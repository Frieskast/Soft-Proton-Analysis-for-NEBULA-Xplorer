#include "detector.hh"
#include "G4AnalysisManager.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4ios.hh"
#include "G4RunManager.hh"
#include "mysteppingaction.hh"
#include "runaction.hh"   
#include "G4Threading.hh"
#include "G4UnitsTable.hh"
#include <unordered_map>
#include <set>

// This function traces back from any track to find the ID of the initial primary particle.
G4int GetPrimaryTrackID(G4Track* aTrack) {
    if (aTrack->GetParentID() == 0) {
        return aTrack->GetTrackID();
    }
    // For secondaries, we need a robust way to find the primary.
    // The most reliable method is to get the primary vertex from the event.
    const G4Event* currentEvent = G4RunManager::GetRunManager()->GetCurrentEvent();
    if (currentEvent) {
        G4PrimaryVertex* primaryVertex = currentEvent->GetPrimaryVertex(0);
        if (primaryVertex) {
            G4PrimaryParticle* primaryParticle = primaryVertex->GetPrimary(0);
            if (primaryParticle) {
                // In Geant4, the primary particle's track ID is always 1 for single-particle events.
                // This is a robust way to get the primary ID.
                return 1; 
            }
        }
    }
    return aTrack->GetTrackID();
}


MySensitiveDetector::MySensitiveDetector(G4String name) : G4VSensitiveDetector(name)
{
}

MySensitiveDetector::~MySensitiveDetector()
{
}

void MySensitiveDetector::Initialize(G4HCofThisEvent*) {
}

// thread-local definitions for static members (one definition per translation unit)
thread_local std::map<G4int, G4double> MySensitiveDetector::siDetKineticEnergies;
thread_local std::map<G4int, G4double> MySensitiveDetector::entryKineticEnergies;
thread_local std::map<G4int, G4double> MySensitiveDetector::lastKineticEnergies;
thread_local std::unordered_map<G4int, G4double> MySensitiveDetector::perEventEdepDet;
thread_local std::unordered_map<G4int, G4double> MySensitiveDetector::perEventEdepFocal;
thread_local std::set<G4int> MySensitiveDetector::s_detailedProtonTrackIDs; // Initialize the new set
thread_local G4int MySensitiveDetector::fDetectorHitCount = 0;
thread_local G4double MySensitiveDetector::total_edep = 0.0;
thread_local G4double MySensitiveDetector::total_edep_det = 0.0;
thread_local G4double MySensitiveDetector::total_edep_focal = 0.0;

// storage for the last printed/merged krad values
double MySensitiveDetector::s_lastDoseKradDet = -1.0;
double MySensitiveDetector::s_lastDoseKradFocal = -1.0;

G4bool MySensitiveDetector::ProcessHits(G4Step *aStep, G4TouchableHistory *ROhist)
{
    // Process all particles for energy deposition (TID), but treat protons specially for
    // entry/exit kinetic-energy bookkeeping and ntuple hit-recording.
    G4Track *track = aStep->GetTrack();
    if (!track) return false;
    auto* pdef = track->GetParticleDefinition();
    if (!pdef) return false;
    const bool isProton = (pdef->GetPDGEncoding() == 2212);

    // Always accumulate deposited energy (all particles contribute to dose).
    G4double edep = aStep->GetTotalEnergyDeposit();
    total_edep += edep;

    G4StepPoint *preStepPoint = aStep->GetPreStepPoint();
    G4StepPoint *postStepPoint = aStep->GetPostStepPoint();
    G4double kineticEnergy = track->GetKineticEnergy();
    
    // --- Declare missing variables at the top of the function ---
    G4int tid = track->GetTrackID();
    G4ThreeVector pos = preStepPoint->GetPosition();

    // Always use the primary's ID as the key for summary maps
    G4int primary_tid = GetPrimaryTrackID(track);

    G4LogicalVolume* lv = preStepPoint->GetTouchableHandle()->GetVolume()->GetLogicalVolume();
    if (!lv) return false;
    G4String lvName = lv->GetName();
    auto analysisManager = G4AnalysisManager::Instance();
    G4int eventID = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();

    // only print / fill on entry to reduce log volume
    bool isEntry = (preStepPoint->GetStepStatus() == fGeomBoundary);

    if (lvName == "SiDetLV") {
        // Record entry/exit kinetic energies only for protons entering/exiting the SiDet volume.
        if (isProton && preStepPoint->GetStepStatus() == fGeomBoundary) {
            // Use the primary track ID as the key
            entryKineticEnergies[primary_tid] = kineticEnergy;
        }
        if (isProton && postStepPoint && postStepPoint->GetStepStatus() == fGeomBoundary) {
            // Use the primary track ID as the key
            lastKineticEnergies[primary_tid] = postStepPoint->GetKineticEnergy();
        }

        // per-event and run-level accumulators (store MeV)
        perEventEdepDet[eventID] += edep / CLHEP::MeV;
        if (const MyRunAction* mra = dynamic_cast<const MyRunAction*>(G4RunManager::GetRunManager()->GetUserRunAction())) {
            MyRunAction* ra = const_cast<MyRunAction*>(mra);
            ra->AddEdepDet(edep / CLHEP::MeV);
        }
        // fill SmallDet ntuple with conditional detail
        if (analysisManager && isProton) {
            bool isMarkedForDetail = s_detailedProtonTrackIDs.count(tid);

            // If this is a new proton entering and we have budget, mark it for detail.
            if (!isMarkedForDetail && isEntry && s_detailedProtonTrackIDs.size() < kMaxDetailedTracks) {
                s_detailedProtonTrackIDs.insert(tid);
                isMarkedForDetail = true;
            }
            

            // Detailed mode: record every step for marked protons.
            if (isMarkedForDetail) {
                if (edep > 0) {
                    analysisManager->FillNtupleDColumn(0, 0, pos.x() / CLHEP::mm);
                    analysisManager->FillNtupleDColumn(0, 1, pos.y() / CLHEP::mm);
                    analysisManager->FillNtupleDColumn(0, 2, pos.z() / CLHEP::mm);
                    analysisManager->FillNtupleDColumn(0, 3, edep / CLHEP::keV); // Store step energy loss
                    analysisManager->FillNtupleDColumn(0, 4, (double)eventID);
                    analysisManager->AddNtupleRow(0);
                }
            }
            // Entry-only mode: for all other protons, just record the entry point and entry energy.
            else if (isEntry) {
                analysisManager->FillNtupleDColumn(0, 0, pos.x() / CLHEP::mm);
                analysisManager->FillNtupleDColumn(0, 1, pos.y() / CLHEP::mm);
                analysisManager->FillNtupleDColumn(0, 2, pos.z() / CLHEP::mm);
                analysisManager->FillNtupleDColumn(0, 3, kineticEnergy / CLHEP::keV); // Store entry kinetic energy
                analysisManager->FillNtupleDColumn(0, 4, (double)eventID);
                analysisManager->AddNtupleRow(0);
            }
        }
    }
    else if (lvName == "DetectorLV") {
        // always add step edep (done below earlier in function); on entry also
        // add the particle's remaining kinetic energy and kill the track so the
        // particle is absorbed by the focal detector.
        perEventEdepFocal[eventID] += edep / CLHEP::MeV;
        const MyRunAction* mra = dynamic_cast<const MyRunAction*>(G4RunManager::GetRunManager()->GetUserRunAction());
        if (mra) {
            MyRunAction* ra = const_cast<MyRunAction*>(mra);
            ra->AddEdepFocal(edep / CLHEP::MeV);
        }

        if (isEntry) {
            // record entry position for any particle
            if (analysisManager) {
                G4double kin = track->GetKineticEnergy();
                analysisManager->FillNtupleDColumn(1, 0, pos.x() / CLHEP::mm);
                analysisManager->FillNtupleDColumn(1, 1, pos.y() / CLHEP::mm);
                analysisManager->FillNtupleDColumn(1, 2, pos.z() / CLHEP::mm);
                analysisManager->FillNtupleDColumn(1, 3, kin / CLHEP::keV); // Fill Ekin column
                analysisManager->FillNtupleDColumn(1, 4, (double)eventID);

                // --- MODIFICATION: Get reflection info and fill material columns ---
                std::string refl1_mat = "";
                std::string refl2_mat = "";
                std::string refl3_mat = "";

                // --- REVERTED: Use G4RunManager to get the stepping action ---
                const auto* steppingAction = static_cast<const MySteppingAction*>(
                    G4RunManager::GetRunManager()->GetUserSteppingAction());

                if (steppingAction) {
                    // Find the reflection history for the current track's primary ancestor
                    G4int primaryTrackID = GetPrimaryTrackID(track);
                    
                    // --- MODIFICATION: Use the new unified trackMirrorHits map ---
                    const auto& hitMap = steppingAction->trackMirrorHits;
                    auto it = hitMap.find(primaryTrackID);

                    if (it != hitMap.end()) {
                        const auto& hits = it->second;
                        if (!hits.empty()) {
                            refl1_mat = hits[0].materialName;
                        }
                        if (hits.size() >= 2) {
                            refl2_mat = hits[1].materialName;
                        }
                        if (hits.size() >= 3) {
                            refl3_mat = hits[2].materialName;
                        }
                    }
                }

                analysisManager->FillNtupleSColumn(1, 5, refl1_mat);
                analysisManager->FillNtupleSColumn(1, 6, refl2_mat);
                analysisManager->FillNtupleSColumn(1, 7, refl3_mat);
                analysisManager->AddNtupleRow(1);
            }

            // add remaining kinetic energy carried by the track to edep accumulators
            G4double kin = track->GetKineticEnergy();
            if (kin > 0.0) {
                total_edep += kin; // keep global total consistent
                perEventEdepFocal[eventID] += kin / CLHEP::MeV;
            if (mra) {
                MyRunAction* ra = const_cast<MyRunAction*>(mra);
                    ra->AddEdepFocal(kin / CLHEP::MeV);
                }
            }

            // terminate the particle immediately (absorbed by focal detector)
            track->SetTrackStatus(fStopAndKill);
        }
    }

    fDetectorHitCount++;
    return true;
}

void MySensitiveDetector::EndOfEvent(G4HCofThisEvent*) {
    // Nothing needed per event
}

void MySensitiveDetector::PrintTIDinKrad() {
    // Only print merged results on master; run accumulables are merged in EndOfRunAction
    if (!G4Threading::IsMasterThread()) {
        G4cout << "PrintTIDinKrad: called on worker — skipping merged values." << G4endl;
        return;
    }

    const MyRunAction* mra = dynamic_cast<const MyRunAction*>(G4RunManager::GetRunManager()->GetUserRunAction());
    if (!mra) {
        G4cerr << "PrintTIDinKrad: MyRunAction not found." << G4endl;
        return;
    }

    // accumulables store edep in MeV
    double edep_det_MeV = mra->GetEdepDet();
    double edep_focal_MeV = mra->GetEdepFocal();
    const double MeV_to_J = 1.602176634e-13;
    double total_edep_J_det = edep_det_MeV * MeV_to_J;
    double total_edep_J_focal = edep_focal_MeV * MeV_to_J;

    auto* logicSiMass = G4LogicalVolumeStore::GetInstance()->GetVolume("SiMassLV"); // whole silicon mass
    auto* logicFocal = G4LogicalVolumeStore::GetInstance()->GetVolume("DetectorLV");
    if (!logicSiMass || !logicFocal) {
        G4cerr << "SiMassLV or DetectorLV not found!" << G4endl;
        return;
    }

    // compute mass using Geant4 internal units: mass_internal = volume_internal * density_internal
    G4Material* matSi = logicSiMass->GetMaterial();
    G4VSolid* solidSi = logicSiMass->GetSolid();
    G4double vol_si_internal = solidSi ? solidSi->GetCubicVolume() : 0.0;
    G4double dens_si_internal = matSi ? matSi->GetDensity() : 0.0;
    G4double mass_si_internal = vol_si_internal * dens_si_internal; // internal mass units

    G4Material* matF = logicFocal->GetMaterial();
    G4VSolid* solidF = logicFocal->GetSolid();
    G4double vol_f_internal = solidF ? solidF->GetCubicVolume() : 0.0;
    G4double dens_f_internal = matF ? matF->GetDensity() : 0.0;
    G4double mass_f_internal = vol_f_internal * dens_f_internal; // internal mass units

    // Convert internal mass units to kilograms for dose calculation
    double mass_si_kg = mass_si_internal / CLHEP::kg;
    double mass_focal_kg = mass_f_internal / CLHEP::kg;

    // compute dose in Gy and krad using SI masses
    double dose_Gy_det = (mass_si_kg > 0.0) ? (total_edep_J_det / mass_si_kg) : 0.0;
    double dose_Gy_focal = (mass_focal_kg > 0.0) ? (total_edep_J_focal / mass_focal_kg) : 0.0;
    double dose_krad_det = dose_Gy_det / 10.0;
    double dose_krad_focal = dose_Gy_focal / 10.0;

    // store results so RunAction can read exactly the printed numbers
    s_lastDoseKradDet = dose_krad_det;
    s_lastDoseKradFocal = dose_krad_focal;

    // G4cout << "PrintTIDinKrad (merged): det_krad=" << dose_krad_det
    //        << " focal_krad=" << dose_krad_focal << G4endl;

    // Do not fill analysis here. TID ntuple filling is performed in RunAction::EndOfRunAction
    // (master-only) to avoid AnalysisManager/ntuple id mismatch across threads.
}

// getters
double MySensitiveDetector::GetLastDoseKradDet() { return s_lastDoseKradDet; }
double MySensitiveDetector::GetLastDoseKradFocal() { return s_lastDoseKradFocal; }
