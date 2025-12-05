#include "runaction.hh"
#include "G4AnalysisManager.hh"
#include "G4RunManager.hh"
#include "G4AccumulableManager.hh"
#include "G4SystemOfUnits.hh"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/resource.h>
#include "generator.hh"
#include "TParameter.h"
#include <TFile.h>
#include "G4EmParameters.hh"
#include <TROOT.h>
#include "G4Run.hh"
#include "detector.hh" // Add this include at the top if not present
#include "G4Threading.hh"
#include "G4UnitsTable.hh"
#include "construction.hh" // Make sure this is at the top of your file
#include "physics.hh"
#include "G4LogicalVolumeStore.hh"

MyRunAction::MyRunAction() : G4UserRunAction(), fExitCount(0), fReflectedCount(0), fReflectedHitCount(0),
    fEdepDet(0.0), fEdepFocal(0.0)
{
    // Register accumulables
    auto mgr = G4AccumulableManager::Instance();
    mgr->Register(fExitCount);
    mgr->Register(fReflectedCount);
    mgr->Register(fReflectedHitCount);
    mgr->Register(fEdepDet);
    mgr->Register(fEdepFocal);

    auto analysisManager = G4AnalysisManager::Instance();

    // Enable ntuple merging for multi-threaded mode
    analysisManager->SetNtupleMerging(true);

    // Set default file type (optional, e.g., root, csv, xml)
    analysisManager->SetDefaultFileType("root");

    // Remove histogram creation
    // analysisManager->CreateH2(...);
    // analysisManager->CreateH1(...);

    // Small detector ntuple (ntuple index 0)
    // Columns:
    //  0: x      (double)  mm
    //  1: y      (double)  mm
    //  2: z      (double)  mm
    //  3: Ekin   (double)  keV
    //  4: EventID(double)  event number
    //  5: trajectory (string) serialized per-event trajectory (if filled)
    //  6: refl_x (double) mm  (reflection point x, optional)
    //  7: refl_y (double) mm  (reflection point y, optional)
    //  8: refl_z (double) mm  (reflection point z, optional)
    //  9: refl_mat (string)   material name at reflection (optional)
    analysisManager->CreateNtuple("SmallDet", "Small Detector Hits");
    analysisManager->CreateNtupleDColumn("x"); // mm
    analysisManager->CreateNtupleDColumn("y");
    analysisManager->CreateNtupleDColumn("z");
    analysisManager->CreateNtupleDColumn("Ekin"); // keV
    analysisManager->CreateNtupleDColumn("EventID");
    analysisManager->CreateNtupleSColumn("trajectory"); // store xyz track as string
    analysisManager->CreateNtupleDColumn("refl_x");
    analysisManager->CreateNtupleDColumn("refl_y");
    analysisManager->CreateNtupleDColumn("refl_z");
    analysisManager->CreateNtupleSColumn("refl_mat");
    analysisManager->FinishNtuple();

    // Focal plane ntuple (ntuple index 1)
    // Columns:
    //  0: x (mm)
    //  1: y (mm)
    //  2: z (mm)
    //  3: Ekin (keV)
    //  4: EventID
    analysisManager->CreateNtuple("FocalDet", "Focal Plane Hits");
    analysisManager->CreateNtupleDColumn("x");
    analysisManager->CreateNtupleDColumn("y");
    analysisManager->CreateNtupleDColumn("z");
    analysisManager->CreateNtupleDColumn("Ekin");
    analysisManager->CreateNtupleDColumn("EventID");
    // --- MODIFICATION: Add columns for reflection materials ---
    analysisManager->CreateNtupleSColumn("refl1_mat");
    analysisManager->CreateNtupleSColumn("refl2_mat");
    analysisManager->CreateNtupleSColumn("refl3_mat");
    analysisManager->FinishNtuple();

    // TID run-level ntuple (ntuple index 2)
    // Columns:
    //  0: TID_krad_det   (double) total ionizing dose in small detector [krad]
    //  1: TID_krad_focal (double) total ionizing dose in focal plane [krad]
    analysisManager->CreateNtuple("TID", "Total Ionizing Dose");
    analysisManager->CreateNtupleDColumn("TID_krad_det");
    analysisManager->CreateNtupleDColumn("TID_krad_focal");
    analysisManager->FinishNtuple();

    // Per-event SmallDet summary (ntuple index 3)
    // One row per event summarizing the primary trajectory and energy through the small detector:
    //  0: EventID
    //  1: trajectory (string) serialized "x,y,z;..."
    //  2: entryE (double) keV  -- kinetic energy at entry to SiDet (keV)
    //  3: exitE  (double) keV  -- kinetic energy at exit from SiDet (keV)
    //  4: lossE  (double) keV  -- entryE - exitE
    //  5: nReflections (int)
    //  6: refl1_x, 7: refl1_y, 8: refl1_z (mm)
    //  9: refl1_mat (string)
    // 10: refl2_x, 11: refl2_y, 12: refl2_z (mm)
    // 13: refl2_mat (string)
    analysisManager->CreateNtuple("SmallDetSummary", "Small Detector Event Summary");
    analysisManager->CreateNtupleDColumn("EventID");
    analysisManager->CreateNtupleSColumn("trajectory");
    analysisManager->CreateNtupleDColumn("entryE");
    analysisManager->CreateNtupleDColumn("exitE");
    analysisManager->CreateNtupleDColumn("lossE");
    analysisManager->CreateNtupleIColumn("nMirrorHits");
    analysisManager->CreateNtupleDColumn("hit1_x");
    analysisManager->CreateNtupleDColumn("hit1_y");
    analysisManager->CreateNtupleDColumn("hit1_z");
    analysisManager->CreateNtupleSColumn("hit1_mat");
    analysisManager->CreateNtupleSColumn("hit1_vol"); // <-- ADD THIS
    analysisManager->CreateNtupleDColumn("hit2_x");
    analysisManager->CreateNtupleDColumn("hit2_y");
    analysisManager->CreateNtupleDColumn("hit2_z");
    analysisManager->CreateNtupleSColumn("hit2_mat");
    analysisManager->CreateNtupleSColumn("hit2_vol"); // <-- ADD THIS
    // --- ADDITION: Columns for the third hit ---
    analysisManager->CreateNtupleDColumn("hit3_x");
    analysisManager->CreateNtupleDColumn("hit3_y");
    analysisManager->CreateNtupleDColumn("hit3_z");
    analysisManager->CreateNtupleSColumn("hit3_mat");
    analysisManager->CreateNtupleSColumn("hit3_vol");
    analysisManager->FinishNtuple(); // ntupleID = 3
}

MyRunAction::~MyRunAction() {}

void MyRunAction::BeginOfRunAction(const G4Run* run) {
    if (G4Threading::IsMasterThread()) {
    fStartTime = std::chrono::high_resolution_clock::now();
    }

    fNEvents = run->GetNumberOfEventToBeProcessed();

        G4AccumulableManager::Instance()->Reset();

    // Ensure the directory exists
    system("mkdir -p build/root");

    // Use run ID to make filename unique
    std::ostringstream oss;
    oss << "build/root/output_pending_run" << run->GetRunID() << ".root";
    G4String filename = oss.str();

    // Set mirror config here, after detector is fully constructed and macro processed
        const G4VUserDetectorConstruction* baseDetector =
            G4RunManager::GetRunManager()->GetUserDetectorConstruction();
        auto* detector = dynamic_cast<MyDetectorConstruction*>(
            const_cast<G4VUserDetectorConstruction*>(baseDetector)
        );
        if (detector) {
            SetMirrorConfig(detector->GetMirrorTypeString());
            SetFilterOn(detector->GetThermalFilterOn());
    }

    // Get the current gun energy from the generator
        auto* basePrimary = const_cast<G4VUserPrimaryGeneratorAction*>(
            G4RunManager::GetRunManager()->GetUserPrimaryGeneratorAction());
            auto* userPrimary = dynamic_cast<MyPrimaryGenerator*>(basePrimary);
            if (userPrimary) {
        SetEnergy(userPrimary->GetParticleGun()->GetParticleEnergy() / CLHEP::keV);
        }

        G4AnalysisManager *man = G4AnalysisManager::Instance();
        man->OpenFile(filename);

    // Store for later renaming
        fPendingFileName = filename;
}

void MyRunAction::EndOfRunAction(const G4Run*) {
    if (G4Threading::IsMasterThread()) {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - fStartTime).count();
        G4cout << "Total run time (all threads): " << duration << " seconds." << G4endl;
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        G4cout << "Memory usage (max RSS): " << usage.ru_maxrss << " KB" << G4endl;
    }

    G4AccumulableManager::Instance()->Merge();

    // Print thread-local TID info (call detector's printer) on master for debugging
    if (G4Threading::IsMasterThread()) {
        MySensitiveDetector::PrintTIDinKrad();  // will print "yo" and the thread-local numbers
    }

    G4AnalysisManager *man = G4AnalysisManager::Instance();

        man->Write();
        man->CloseFile();

        // Rename the file to include the correct angle and energy
        std::string oldName = fPendingFileName;
        std::string newName = "build/root/" + GenerateFileName();
        std::rename(oldName.c_str(), newName.c_str());
    }

G4String MyRunAction::GenerateFileName() {
    double thetaLimit = G4EmParameters::Instance()->MscThetaLimit();
    std::ostringstream oss;
    oss << "output_"
        << (fMirrorConfig.empty() ? "unknown" : fMirrorConfig) << "_"
        << (fFilterOn ? "filterOn_" : "filterOff_")
        << fEnergyKeV << "keV_"
        << GetSelectedPhysics() << "_"    // <-- inserted physics token
        << "N";
    if (fNEvents >= 1000000) {
        oss << (fNEvents / 1000000) << "m";
    } else if (fNEvents >= 1000) {
        oss << (fNEvents / 1000) << "k";
    } else {
        oss << fNEvents;
    }
    oss << ".root";
    return oss.str();
}




