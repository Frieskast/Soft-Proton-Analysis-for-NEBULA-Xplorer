#include "runaction.hh"
#include "G4AnalysisManager.hh"
#include "G4RunManager.hh"
#include "G4AccumulableManager.hh"
#include "G4SystemOfUnits.hh"
#include <fstream>
#include <sstream>
#include <iomanip>
#include "generator.hh"
#include "TParameter.h"
#include <TFile.h>
#include "G4EmParameters.hh"
#include <TROOT.h>
#include "G4Run.hh"
#include <sys/resource.h> // For memory monitoring

// New includes for run.mac parsing and safe directory creation
#include <filesystem>
#include <regex>
#include <vector>
#include <system_error>
// Timer
#include <chrono>
#include "G4Threading.hh"

// file-scope timer for the current run (no header change required)
namespace {
    std::chrono::steady_clock::time_point gRunStart;
}

MyRunAction::MyRunAction() : G4UserRunAction(), fExitCount(0) {
    G4AccumulableManager::Instance()->Register(fExitCount);

    auto man = G4AnalysisManager::Instance();

    // Enable ntuple merging for multi-threaded mode
    man->SetNtupleMerging(true);

    // Set default file type
    man->SetDefaultFileType("root");

    man->CreateH1("Energy", "Exit/Incident Energy", 40, 0.0, 1.0);
    man->CreateH1("Scattering angle 0-4 deg", "Scattering angle 0-4.5", 45, 0.0, 4.5);
    man->CreateH1("Scattering angle 4-10 deg", "Scattering angle 4.5-10", 26, 4.5, 10.0);

    // Create metadata ntuple
    man->CreateNtuple("MetaData", "Run metadata");
    man->CreateNtupleDColumn("NumIncidentProtons");
    man->FinishNtuple();

    // Create scattering events ntuple (for individual event analysis)
    man->CreateNtuple("ScatterEvents", "Scattering data");
    man->CreateNtupleDColumn("theta_deg");       // Scattering angle
    man->CreateNtupleDColumn("energy_keV");      // Energy after reflection       
    man->CreateNtupleDColumn("x_mm");            // Impact position
    man->CreateNtupleDColumn("y_mm");
    man->CreateNtupleDColumn("z_mm");
    man->FinishNtuple();
}

MyRunAction::~MyRunAction() {}

void MyRunAction::BeginOfRunAction(const G4Run* run) {
    fNEvents = run->GetNumberOfEventToBeProcessed();
    G4AccumulableManager::Instance()->Reset();

    // start run timer
    gRunStart = std::chrono::steady_clock::now();
    
    // Parse run.mac to pick the energy and incidentAngle active for this /run/beamOn
    // The file may contain multiple /run/beamOn entries; targetCount is 1-based index.
    double parsedEnergyKeV = fEnergyKeV; // keep existing if preset
    double parsedAngleDeg = fIncidentAngle / CLHEP::deg; // degrees

    std::ifstream mac("run.mac");
    if (mac) {
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(mac, line)) {
            auto first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) { lines.emplace_back(""); continue; }
            auto last = line.find_last_not_of(" \t\r\n");
            lines.emplace_back(line.substr(first, last - first + 1));
        }

        std::regex re_energy(R"(^\s*/gun/energy\s+([0-9.+-eE]+)\s*(keV|MeV|eV)?)", std::regex::icase);
        std::regex re_angle(R"(^\s*/gun/incidentAngle\s+([0-9.+-eE]+))", std::regex::icase);
        std::regex re_beamon(R"(^\s*/run/beamOn\b)", std::regex::icase);
        std::smatch m;

        int targetCount = 1 + (run ? run->GetRunID() : 0); // 1-based beamOn index
        int beamonCount = 0;

        double lastEnergy = parsedEnergyKeV;   // keV
        double lastAngleDeg = parsedAngleDeg;  // degrees

        for (const auto &ln : lines) {
            if (std::regex_search(ln, m, re_energy)) {
                double val = std::stod(m[1].str());
                std::string unit = m[2].matched ? m[2].str() : "keV";
                if (unit == "MeV" || unit == "mev") val *= 1000.0;
                else if (unit == "eV" || unit == "EV") val *= 1.0e-3;
                lastEnergy = val;
                continue;
            }
            if (std::regex_search(ln, m, re_angle)) {
                double val = std::stod(m[1].str());
                lastAngleDeg = val; // assume macro uses degrees
                continue;
            }
            if (std::regex_search(ln, m, re_beamon)) {
                ++beamonCount;
                if (beamonCount == targetCount) {
                    // adopt last seen values
                    if (lastEnergy > 0.0) {
                        parsedEnergyKeV = lastEnergy;
                    }
                    if (lastAngleDeg > 0.0) {
                        parsedAngleDeg = lastAngleDeg;
                    }
                    break;
                }
            }
        }
    }

    // Commit parsed values into run-action members (convert angle to radians)
    if (parsedEnergyKeV > 0.0) {
        fEnergyKeV = parsedEnergyKeV;
        fIncidentEnergy = parsedEnergyKeV * CLHEP::keV;
    }
    if (parsedAngleDeg > 0.0) {
        fIncidentAngle = parsedAngleDeg * CLHEP::deg;
    }

    // Ensure output directory exists (avoid fork/system)
    std::error_code ec;
    std::filesystem::create_directories("build/root", ec);
    if (ec) {
        G4cerr << "Could not create directory build/root: " << ec.message() << G4endl;
    }

    // Build final filename now (per-run) and open it directly so each /run/beamOn creates a new file
    G4String finalName = GenerateFileName(run);
    std::string fullpath = std::string("build/root/") + std::string(finalName);
    G4AnalysisManager *man = G4AnalysisManager::Instance();
    man->OpenFile(fullpath);
    fPendingFileName = fullpath;
    G4cout << "Opened file: " << fullpath << G4endl;
}

void MyRunAction::EndOfRunAction(const G4Run* run) {
    G4AccumulableManager::Instance()->Merge();

    G4AnalysisManager *man = G4AnalysisManager::Instance();

    // Fill the metadata ntuple with the number of incident protons
    // man->FillNtupleDColumn(0, 0, fIncidentCount); // Fill NumIncidentProtons
    man->AddNtupleRow(0); // Add the row to the metadata ntuple

    man->Write();
    man->CloseFile();

    // Print memory usage and elapsed wall time once on master thread
    if (G4Threading::IsMasterThread()) {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        G4cout << "Memory usage (max RSS): " << usage.ru_maxrss << " KB" << G4endl;

        // elapsed time
        auto now = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration_cast<std::chrono::duration<double>>(now - gRunStart).count();
        G4cout << "Run elapsed time: " << std::fixed << std::setprecision(3) << elapsed_s << " s" << G4endl;

        // --- Append run info to a single metadata file ---
        std::ofstream meta("build/root/all_run_metadata.txt", std::ios::app);
        if (meta) {
            // Gather info
            int runID = run ? run->GetRunID() : -1;
            double energy_keV = fEnergyKeV > 0.0 ? fEnergyKeV : (fIncidentEnergy / CLHEP::keV);
            double angle_deg = fIncidentAngle / CLHEP::deg;
            meta << std::fixed << std::setprecision(3)
                 << "runID=" << runID
                 << "  nEvents=" << fNEvents
                 << "  elapsed_s=" << elapsed_s
                 << "  energy_keV=" << energy_keV
                 << "  angle_deg=" << angle_deg
                 << "  file=" << fPendingFileName
                 << std::endl;
        } else {
            G4cerr << "Could not write to build/root/all_run_metadata.txt" << G4endl;
        }
    }
}
 

G4String MyRunAction::GenerateFileName(const G4Run* run) {
    // Use stored run-action members as primary source
    double angleDeg = fIncidentAngle / CLHEP::deg; // may be 0 if not set
    double energyKeV = fEnergyKeV;                 // scalar keV
    if (energyKeV <= 0.0 && fIncidentEnergy > 0.0) {
        energyKeV = fIncidentEnergy / CLHEP::keV;
    }

    // If run provided a number of events, prefer that
    G4int nEvents = fNEvents;
    G4int runID = 0;
    if (run) {
        G4int runN = run->GetNumberOfEventToBeProcessed();
        if (runN > 0) nEvents = runN;
        runID = run->GetRunID(); // unique per /run/beamOn
    }

    double thetaLimit = G4EmParameters::Instance()->MscThetaLimit();

    std::ostringstream oss;
    oss << "output_"
        << std::fixed << std::setprecision(2) << angleDeg << "deg_"
        << static_cast<int>(std::round(energyKeV)) << "keV_"
        << "N";
    if (nEvents >= 1000000) {
        // show millions, with one decimal if not an integer million (e.g. 1.5m)
        if (nEvents % 1000000 == 0) {
            oss << (nEvents / 1000000) << "m";
        } else if (nEvents % 100000 == 0) {
            double v = static_cast<double>(nEvents) / 1e6;
            oss << std::fixed << std::setprecision(1) << v << "m";
        } else {
            double v = static_cast<double>(nEvents) / 1e6;
            oss << std::fixed << std::setprecision(2) << v << "m";
        }
    } else if (nEvents >= 1000) {
        // show thousands, with one decimal if not an integer thousand (e.g. 1.5k)
        if (nEvents % 1000 == 0) {
            oss << (nEvents / 1000) << "k";
        } else if (nEvents % 100 == 0) {
            double v = static_cast<double>(nEvents) / 1e3;
            oss << std::fixed << std::setprecision(1) << v << "k";
        } else {
            double v = static_cast<double>(nEvents) / 1e3;
            oss << std::fixed << std::setprecision(2) << v << "k";
        }
    } else {
        oss << nEvents;
    }

    // append run id to force uniqueness per beamOn
    oss << "_r" << runID << ".root";

    return G4String(oss.str());
}



