#include "generator.hh"
#include "G4Event.hh" // Include for G4Event
#include "Randomize.hh" // Include for G4UniformRand
#include "G4SystemOfUnits.hh" // For unit definitions
#include "G4ProductionCutsTable.hh" // Include for production cuts table
#include "G4RegionStore.hh" // Include for region store
#include "G4ProductionCuts.hh" // Include for production cuts
#include "runaction.hh"
#include "G4RunManager.hh"
#include "generatorMessenger.hh"

#include <fstream>
#include <sstream>
#include <map>
#include <cmath>

namespace {
    // load CSV once into a map: key = (energy_keV, angle_centideg) -> sigma_deg
    static std::map<std::pair<int,int>, double> LoadAngleSigmaMap(const std::string& csvpath="all_efficiencies_combined.csv") {
        std::map<std::pair<int,int>, double> m;
        std::ifstream ifs(csvpath);
        if (!ifs.is_open()) {
            G4cout << "Warning: CSV file not found: " << csvpath << "  (no angular smearing will be applied)" << G4endl;
            return m;
        }
        std::string line;
        // skip header
        std::getline(ifs, line);
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string token;
            // CSV columns:
            // energy_keV,inc_ang_deg,inc_ang_err_deg,scat_ang_deg,scat_ang_err_deg,efficiency_sr_inv,efficiency_err_sr_inv
            int energy = 0;
            double inc_ang = 0.0;
            double inc_ang_err = 0.0;
            // read fields
            if (!std::getline(ss, token, ',')) continue;
            energy = static_cast<int>(std::round(std::stod(token)));
            if (!std::getline(ss, token, ',')) continue;
            inc_ang = std::stod(token);
            if (!std::getline(ss, token, ',')) continue;
            inc_ang_err = std::stod(token);
            // fold into map keyed by energy and centi-degrees (two decimals)
            int ang_key = static_cast<int>(std::round(inc_ang * 100.0));
            m[std::make_pair(energy, ang_key)] = inc_ang_err;
        }
        G4cout << "Loaded " << m.size() << " angle-sigma entries from CSV" << G4endl;
        return m;
    }

    static const std::map<std::pair<int,int>, double>& AngleSigmaMap() {
        static const auto map_inst = LoadAngleSigmaMap();
        return map_inst;
    }
}

MyPrimaryGenerator::MyPrimaryGenerator()
{
    fParticleGun = new G4ParticleGun(1); // Initialize particle gun with 1 particle per event

    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();

    // Generate a proton
    G4String particleName = "proton";
    G4ParticleDefinition* particle = particleTable->FindParticle(particleName);

    // Set the particle energy
    fParticleGun->SetParticleEnergy(1000 * keV); // Example energy

    // Set the particle definition
    fParticleGun->SetParticleDefinition(particle);

    fMessenger = new GeneratorMessenger(this);
}

MyPrimaryGenerator::~MyPrimaryGenerator()
{
    delete fMessenger;
    delete fParticleGun;
}

void MyPrimaryGenerator::GeneratePrimaries(G4Event* anEvent) {
    // Convert the incident angle from stored units to radians/CLHEP units
    // fIncidentAngle is expected to be provided in Geant4 angular units (e.g. degrees * CLHEP::deg).
    G4double incidentAngleRad = fIncidentAngle; // keep in CLHEP units

    // Use a large source distance so smeared rays still hit the target
    const G4double sourceDist = 0.1 * CLHEP::m; // increase if needed
    // nominal (unsmeared) momentum direction
    G4double xMom = 0.0;
    G4double yMom = -std::sin(incidentAngleRad);
    G4double zMom = -std::cos(incidentAngleRad);
    G4ThreeVector mom(xMom, yMom, zMom);
    mom = mom.unit();

    // place source opposite to momentum at distance sourceDist
    G4ThreeVector sourcePos = - sourceDist * mom;
    fParticleGun->SetParticlePosition(sourcePos);

    // apply angular smearing (sigma_deg from CSV already computed earlier)
    double sigma_deg = 0.0;
    {
        const auto &mmap = AngleSigmaMap();
        // get energy in keV (round)
        double energy_keV_d = fParticleGun->GetParticleEnergy() / keV;
        int energy_key = static_cast<int>(std::round(energy_keV_d));
        // incident angle in degrees (two-decimal key)
        double inc_ang_deg = incidentAngleRad / CLHEP::deg;
        int ang_key = static_cast<int>(std::round(inc_ang_deg * 100.0));
        auto it = mmap.find(std::make_pair(energy_key, ang_key));
        if (it != mmap.end()) {
            sigma_deg = it->second;
        } else {
            // no exact match: try to find any entry with same energy and nearest angle (tolerance)
            int best_key = 0;
            double best_diff = 1e9;
            for (const auto &p : mmap) {
                if (p.first.first != energy_key) continue;
                double adeg = p.first.second / 100.0;
                double diff = std::abs(adeg - inc_ang_deg);
                if (diff < best_diff) {
                    best_diff = diff;
                    best_key = p.first.second;
                }
            }
            if (best_diff < 0.5) { // within 0.5 deg tolerance
                auto jt = mmap.find(std::make_pair(energy_key, best_key));
                if (jt != mmap.end()) sigma_deg = jt->second;
            }
        }
    }

    if (sigma_deg <= 0.0) {
        fParticleGun->SetParticleMomentumDirection(mom);
    } else {
        G4double sigma_rad = sigma_deg * CLHEP::deg;
        // build orthonormal basis perpendicular to mom
        G4ThreeVector ux = mom.orthogonal().unit();
        G4ThreeVector uy = mom.cross(ux).unit();
        G4double dx = G4RandGauss::shoot(0.0, sigma_rad);
        G4double dy = G4RandGauss::shoot(0.0, sigma_rad);
        G4ThreeVector newDir = (mom + dx * ux + dy * uy).unit();
        fParticleGun->SetParticleMomentumDirection(newDir);

        // debug: print a few sampled primaries to ensure they aim at target
        static G4int dbgCount = 0;
        if (dbgCount < 10) {
            G4cout << "GEN_DBG: pos=" << sourcePos << " dir_nom=" << mom << " dir_smeared=" << newDir
                   << " sigma_deg=" << sigma_deg << G4endl;
            ++dbgCount;
        }
    }

    // Generate the primary vertex
    fParticleGun->GeneratePrimaryVertex(anEvent);

    // Do not update MyRunAction here (run-action API was simplified).
}

void MyPrimaryGenerator::SetEnergy(G4double energy)
{
    // energy is in Geant4 internal units (value returned by UI cmd GetNewDoubleValue)
    fParticleGun->SetParticleEnergy(energy);
}

