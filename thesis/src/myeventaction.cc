#include "myeventaction.hh"
#include "G4Event.hh"
#include "G4TrajectoryContainer.hh"
#include "G4Trajectory.hh"
#include "G4AnalysisManager.hh"
#include "CLHEP/Units/SystemOfUnits.h"
#include <cmath>
#include <sstream>
#include "detector.hh" // to access MySensitiveDetector static maps if available
#include "mysteppingaction.hh"
#include "G4RunManager.hh"
#include "G4TrajectoryPoint.hh"
#include "G4UImanager.hh"
#include <atomic>
#include "runaction.hh"   // make MyRunAction visible so we can cast to it
#include "G4LogicalVolumeStore.hh"
#include <iomanip>
#include <limits>

// --- Implement new constructor ---
MyEventAction::MyEventAction(MySteppingAction* steppingAction) 
: G4UserEventAction(), fSteppingAction(steppingAction) 
{}

MyEventAction::~MyEventAction() {}

// thread-safe counter for saved trajectories (shared across threads)
static std::atomic<int> g_savedTrajCount{0};
static constexpr int kMaxSavedTraj = 100;

void MyEventAction::BeginOfEventAction(const G4Event* /*event*/) {
    // --- Clear the new unified map ---
    if (fSteppingAction) {
        fSteppingAction->trackMirrorHits.clear();
        fSteppingAction->trackTrajectories.clear();
        fSteppingAction->trackEnteredMirrors.clear();
    }

    // Clear per-event / per-track maps in the sensitive detector to avoid cross-event leakage
    MySensitiveDetector::entryKineticEnergies.clear();
    MySensitiveDetector::lastKineticEnergies.clear();

    // BeginOfEventAction already clears MySensitiveDetector maps — also clear perEvent maps in detector:
    MySensitiveDetector::perEventEdepDet.clear();
    MySensitiveDetector::perEventEdepFocal.clear();
}

void MyEventAction::EndOfEventAction(const G4Event* event) {
    auto analysisManager = G4AnalysisManager::Instance();
    G4int evtID = event->GetEventID();

    // Get trajectory container from the event and collect trajectories of protons that hit the small detector.
    G4TrajectoryContainer* trajContainer = event->GetTrajectoryContainer();
    std::string traj_str;                       // will contain concatenated proton trajectories hitting SiDet
    G4ThreeVector last(0,0,0);
    G4int primaryTrackID = -1;
    std::vector<G4ThreeVector> points; // moved out so visible below when deciding saveTrajectory
    std::vector<std::string> savedTrajStrings; // declare here so visible after trajContainer block

    if (trajContainer) {
        // First, find the primary track id (track with ParentID==0 or trackID==1) so we only
        // collect/serialize the primary trajectory (avoids collecting many secondaries).
        for (int i = 0; i < trajContainer->entries(); ++i) {
            G4VTrajectory* vtrj = static_cast<G4VTrajectory*>((*trajContainer)[i]);
            G4Trajectory* traj = dynamic_cast<G4Trajectory*>(vtrj);
            if (!traj) continue;
            if (traj->GetTrackID() == 1 || traj->GetParentID() == 0) { primaryTrackID = traj->GetTrackID(); break; }
        }

        if (primaryTrackID > 0) {
            // collect only the primary trajectory (or fail gracefully)
            for (int i = 0; i < trajContainer->entries(); ++i) {
                G4VTrajectory* vtrj = static_cast<G4VTrajectory*>((*trajContainer)[i]);
                G4Trajectory* traj = dynamic_cast<G4Trajectory*>(vtrj);
                if (!traj) continue;
                if (traj->GetTrackID() != primaryTrackID) continue;
                if (traj->GetParticleName() != "proton") continue;

                G4int tid = traj->GetTrackID();
                // ensure the primary did hit the SiDet (entry KE known)
                if (!MySensitiveDetector::entryKineticEnergies.count(tid)) continue;

                std::ostringstream oss_tr;
                for (G4int p = 0; p < traj->GetPointEntries(); ++p) {
                    const G4VTrajectoryPoint* tp = traj->GetPoint(p);
                    if (!tp) continue;
                    auto pos = tp->GetPosition();
                    if (p == 0) points.push_back(pos); // remember first point for radius test
                    oss_tr << std::fixed << std::setprecision(2)
                           << pos.x()/CLHEP::mm << "," << pos.y()/CLHEP::mm << "," << pos.z()/CLHEP::mm << ";";
                }
                savedTrajStrings.push_back(oss_tr.str());
                break; // only primary
            }
        }
    }

    // After collecting savedTrajStrings above, make sure traj_str is set
    // prefer the serialized primary trajectory if we collected it
    if (!savedTrajStrings.empty()) {
        traj_str = savedTrajStrings.front();
    }

    // Get entry/exit/loss for the primary if available. Use NaN for missing exit so it
    // is clear in the ntuple that the particle did not exit.
    double entryE_keV = std::numeric_limits<double>::quiet_NaN();
    double exitE_keV  = std::numeric_limits<double>::quiet_NaN();
    double loss_keV   = std::numeric_limits<double>::quiet_NaN();
    if (primaryTrackID > 0) {
        if (MySensitiveDetector::entryKineticEnergies.count(primaryTrackID))
            entryE_keV = MySensitiveDetector::entryKineticEnergies.at(primaryTrackID) / CLHEP::keV;
        if (MySensitiveDetector::lastKineticEnergies.count(primaryTrackID))
            exitE_keV = MySensitiveDetector::lastKineticEnergies.at(primaryTrackID) / CLHEP::keV;
        if (!std::isnan(entryE_keV) && !std::isnan(exitE_keV)) loss_keV = entryE_keV - exitE_keV;
    }

    // Decide whether to store the full trajectory string:
    // - only for primary protons
    // - primary must have an entry record in the SiDet maps (it hit the small detector)
    // - primary's first trajectory point must be outside the small detector active radius
    // - only save the first kMaxSavedTraj (=100) trajectories across all threads
    bool saveTrajectory = false;
    if (primaryTrackID > 0 && MySensitiveDetector::entryKineticEnergies.count(primaryTrackID)) {
        if (!points.empty()) {
            const double detActiveRadiusMM = 4.0; // mm (matches construction.cc det_active_radius)
            double r0_mm = std::sqrt(std::pow(points.front().x()/CLHEP::mm,2) + std::pow(points.front().y()/CLHEP::mm,2));
            if (r0_mm > detActiveRadiusMM) {
                // reserve a slot atomically
                int prev = g_savedTrajCount.fetch_add(1, std::memory_order_relaxed);
                if (prev < kMaxSavedTraj) saveTrajectory = true;
             }
         }
     }

    // In EndOfEventAction, compute per-event totals and primary loss:
    int trackKey = (primaryTrackID > 0) ? primaryTrackID : -1; // -1 indicates no primary
    double event_edep_det_MeV = 0.0;
    double event_edep_focal_MeV = 0.0;
    if (MySensitiveDetector::perEventEdepDet.count(evtID)) event_edep_det_MeV = MySensitiveDetector::perEventEdepDet[evtID];
    if (MySensitiveDetector::perEventEdepFocal.count(evtID)) event_edep_focal_MeV = MySensitiveDetector::perEventEdepFocal[evtID];

    // --- Fill per-event TID ntuple (ntuple index 2) so workers store their local TID rows ---
    {
        auto* man = G4AnalysisManager::Instance();
        if (man) {
            const double MeV_to_J = 1.602176634e-13;

            double total_edep_J_det = event_edep_det_MeV * MeV_to_J;
            double total_edep_J_focal = event_edep_focal_MeV * MeV_to_J;

            double dose_krad_det_event = 0.0;
            double dose_krad_focal_event = 0.0;
            auto* logicSiMass = G4LogicalVolumeStore::GetInstance()->GetVolume("SiMassLV");
            auto* logicFocal = G4LogicalVolumeStore::GetInstance()->GetVolume("DetectorLV");
            if (logicSiMass && logicFocal) {
                G4VSolid* solidSi = logicSiMass->GetSolid();
                G4Material* matSi = logicSiMass->GetMaterial();
                G4double vol_si_internal = solidSi ? solidSi->GetCubicVolume() : 0.0;
                G4double dens_si_internal = matSi ? matSi->GetDensity() : 0.0;
                G4double mass_si_internal = vol_si_internal * dens_si_internal;
                double mass_si_kg = mass_si_internal / CLHEP::kg;
                
                G4VSolid* solidF = logicFocal->GetSolid();
                G4Material* matF = logicFocal->GetMaterial();
                G4double vol_f_internal = solidF ? solidF->GetCubicVolume() : 0.0;
                G4double dens_f_internal = matF ? matF->GetDensity() : 0.0;
                G4double mass_f_internal = vol_f_internal * dens_f_internal;
                double mass_focal_kg = mass_f_internal / CLHEP::kg;

                if (mass_si_kg > 0.0) dose_krad_det_event = (total_edep_J_det / mass_si_kg) / 10.0;
                if (mass_focal_kg > 0.0) dose_krad_focal_event = (total_edep_J_focal / mass_focal_kg) / 10.0;
            }

            // Fill TID ntuple (index 2): column 0 = det krad, column 1 = focal krad
            // Booking must be done on master before workers run (SetNtupleMerging(true) required).
            // Guard: only fill if the local AnalysisManager knows the TID ntuple.
            if (man->GetNtuple("TID") != nullptr) {
                const double tiny_threshold = 0.0;
                if ((dose_krad_det_event > tiny_threshold) || (dose_krad_focal_event > tiny_threshold)) {
                    man->FillNtupleDColumn(2, 0, dose_krad_det_event);
                    man->FillNtupleDColumn(2, 1, dose_krad_focal_event);
                    man->AddNtupleRow(2);
                }
            } 
        } else {
            G4cerr << "MyEventAction::EndOfEventAction: AnalysisManager not available; skipping per-event TID fill." << G4endl;
        }
    } // End of TID ntuple block

    // compute primary loss (keV) if entry and exit known for the primary track
    double primary_loss_keV = NAN;
    if (primaryTrackID > 0) {
        if (MySensitiveDetector::entryKineticEnergies.count(primaryTrackID) &&
            MySensitiveDetector::lastKineticEnergies.count(primaryTrackID)) {
            double entry_keV = MySensitiveDetector::entryKineticEnergies[primaryTrackID] / CLHEP::keV;
            double exit_keV  = MySensitiveDetector::lastKineticEnergies[primaryTrackID] / CLHEP::keV;
            primary_loss_keV = entry_keV - exit_keV;
        }
    }

    // Fill per-event summary ntuple (SmallDetSummary) by getting the booking and its id.
    auto* nb = analysisManager->GetNtuple("SmallDetSummary");
    if (nb == nullptr) {
        G4cerr << "MyEventAction: SmallDetSummary ntuple not found (skipping per-event summary fill)" << G4endl;
    } else {
        // GetNtuple returned a booking object but its internals differ across Geant4 versions.
        // Use the known creation order: SmallDet=0, FocalDet=1, TID=2, SmallDetSummary=3.
        // If your RunAction created SmallDetSummary in a different order, adjust this index.
        G4int summaryNtupleId = 3;
        // hitSmallDet is true only when the primary has an entry recorded in the SiDet maps
        // Strict primary-hit test: require primaryTrackID and an entryKE recorded for that primary.
        // This guarantees one (possible) summary row per event for the primary only.
        bool hitSmallDet = (primaryTrackID > 0) && MySensitiveDetector::entryKineticEnergies.count(primaryTrackID);

         if (hitSmallDet) {
            // --- MODIFICATION: Get hit count from the new unified map ---
            int nMirrorHits = 0;
            if (fSteppingAction && fSteppingAction->trackMirrorHits.count(primaryTrackID)) {
                nMirrorHits = fSteppingAction->trackMirrorHits.at(primaryTrackID).size();
            }

            // --- Calculate energy loss ---
            double entryE_keV = MySensitiveDetector::entryKineticEnergies.at(primaryTrackID) / CLHEP::keV;
            double exitE_keV = 0.0; 
            if (MySensitiveDetector::lastKineticEnergies.count(primaryTrackID)) {
                exitE_keV = MySensitiveDetector::lastKineticEnergies.at(primaryTrackID) / CLHEP::keV;
            }
            double loss_keV = entryE_keV - exitE_keV;

            // --- Fill the ntuple using the new unified logic ---
            analysisManager->FillNtupleDColumn(summaryNtupleId, 0, (double)evtID);
            if (saveTrajectory) analysisManager->FillNtupleSColumn(summaryNtupleId, 1, traj_str);
            else analysisManager->FillNtupleSColumn(summaryNtupleId, 1, std::string());
            analysisManager->FillNtupleDColumn(summaryNtupleId, 2, entryE_keV);
            analysisManager->FillNtupleDColumn(summaryNtupleId, 3, exitE_keV);
            analysisManager->FillNtupleDColumn(summaryNtupleId, 4, loss_keV);
            analysisManager->FillNtupleIColumn(summaryNtupleId, 5, nMirrorHits); // Use nMirrorHits

            // Process hits from the new unified map
            if (nMirrorHits > 0) {
                auto const& hits = fSteppingAction->trackMirrorHits.at(primaryTrackID);
                
                // First hit
                const auto& hit1 = hits[0];
                analysisManager->FillNtupleDColumn(summaryNtupleId, 6, hit1.position.x()/CLHEP::mm);
                analysisManager->FillNtupleDColumn(summaryNtupleId, 7, hit1.position.y()/CLHEP::mm);
                analysisManager->FillNtupleDColumn(summaryNtupleId, 8, hit1.position.z()/CLHEP::mm);
                analysisManager->FillNtupleSColumn(summaryNtupleId, 9, hit1.materialName);
                analysisManager->FillNtupleSColumn(summaryNtupleId, 10, hit1.volumeName); // <-- FILL VOLUME NAME

                // Second hit (if it exists)
                if (nMirrorHits > 1) {
                    const auto& hit2 = hits[1];
                    analysisManager->FillNtupleDColumn(summaryNtupleId, 11, hit2.position.x()/CLHEP::mm);
                    analysisManager->FillNtupleDColumn(summaryNtupleId, 12, hit2.position.y()/CLHEP::mm);
                    analysisManager->FillNtupleDColumn(summaryNtupleId, 13, hit2.position.z()/CLHEP::mm);
                    analysisManager->FillNtupleSColumn(summaryNtupleId, 14, hit2.materialName);
                    analysisManager->FillNtupleSColumn(summaryNtupleId, 15, hit2.volumeName);

                    // --- ADDITION: Third hit (if it exists) ---
                    if (nMirrorHits > 2) {
                        const auto& hit3 = hits[2];
                        analysisManager->FillNtupleDColumn(summaryNtupleId, 16, hit3.position.x()/CLHEP::mm);
                        analysisManager->FillNtupleDColumn(summaryNtupleId, 17, hit3.position.y()/CLHEP::mm);
                        analysisManager->FillNtupleDColumn(summaryNtupleId, 18, hit3.position.z()/CLHEP::mm);
                        analysisManager->FillNtupleSColumn(summaryNtupleId, 19, hit3.materialName);
                        analysisManager->FillNtupleSColumn(summaryNtupleId, 20, hit3.volumeName);
                    } else { // Fill third hit with NaN/empty if it doesn't exist
                        analysisManager->FillNtupleDColumn(summaryNtupleId, 16, std::numeric_limits<double>::quiet_NaN());
                        analysisManager->FillNtupleDColumn(summaryNtupleId, 17, std::numeric_limits<double>::quiet_NaN());
                        analysisManager->FillNtupleDColumn(summaryNtupleId, 18, std::numeric_limits<double>::quiet_NaN());
                        analysisManager->FillNtupleSColumn(summaryNtupleId, 19, "");
                        analysisManager->FillNtupleSColumn(summaryNtupleId, 20, "");
                    }
                } else { // Fill second and third hits with NaN/empty if they don't exist
                    analysisManager->FillNtupleDColumn(summaryNtupleId, 11, std::numeric_limits<double>::quiet_NaN());
                    analysisManager->FillNtupleDColumn(summaryNtupleId, 12, std::numeric_limits<double>::quiet_NaN());
                    analysisManager->FillNtupleDColumn(summaryNtupleId, 13, std::numeric_limits<double>::quiet_NaN());
                    analysisManager->FillNtupleSColumn(summaryNtupleId, 14, "");
                    analysisManager->FillNtupleSColumn(summaryNtupleId, 15, "");
                    analysisManager->FillNtupleDColumn(summaryNtupleId, 16, std::numeric_limits<double>::quiet_NaN());
                    analysisManager->FillNtupleDColumn(summaryNtupleId, 17, std::numeric_limits<double>::quiet_NaN());
                    analysisManager->FillNtupleDColumn(summaryNtupleId, 18, std::numeric_limits<double>::quiet_NaN());
                    analysisManager->FillNtupleSColumn(summaryNtupleId, 19, "");
                    analysisManager->FillNtupleSColumn(summaryNtupleId, 20, "");
                }
            } else { // Fill all hit columns with NaN/empty if no hits
                analysisManager->FillNtupleDColumn(summaryNtupleId, 6, std::numeric_limits<double>::quiet_NaN());
                analysisManager->FillNtupleDColumn(summaryNtupleId, 7, std::numeric_limits<double>::quiet_NaN());
                analysisManager->FillNtupleDColumn(summaryNtupleId, 8, std::numeric_limits<double>::quiet_NaN());
                analysisManager->FillNtupleSColumn(summaryNtupleId, 9, "");
                analysisManager->FillNtupleSColumn(summaryNtupleId, 10, "");
                analysisManager->FillNtupleDColumn(summaryNtupleId, 11, std::numeric_limits<double>::quiet_NaN());
                analysisManager->FillNtupleDColumn(summaryNtupleId, 12, std::numeric_limits<double>::quiet_NaN());
                analysisManager->FillNtupleDColumn(summaryNtupleId, 13, std::numeric_limits<double>::quiet_NaN());
                analysisManager->FillNtupleSColumn(summaryNtupleId, 14, "");
                analysisManager->FillNtupleSColumn(summaryNtupleId, 15, "");
                analysisManager->FillNtupleDColumn(summaryNtupleId, 16, std::numeric_limits<double>::quiet_NaN());
                analysisManager->FillNtupleDColumn(summaryNtupleId, 17, std::numeric_limits<double>::quiet_NaN());
                analysisManager->FillNtupleDColumn(summaryNtupleId, 18, std::numeric_limits<double>::quiet_NaN());
                analysisManager->FillNtupleSColumn(summaryNtupleId, 19, "");
                analysisManager->FillNtupleSColumn(summaryNtupleId, 20, "");
            }

            // The old substrate columns are gone, so we don't fill them.
            analysisManager->AddNtupleRow(summaryNtupleId);
         }
     }

    // NOTE: per-step SmallDet ntuple (ntuple "SmallDet") is filled inside the sensitive detector
    // ProcessHits. We avoid writing a mixed per-event row into that per-step ntuple here to
    // prevent column/type mismatches and confusing mixed rows.
}