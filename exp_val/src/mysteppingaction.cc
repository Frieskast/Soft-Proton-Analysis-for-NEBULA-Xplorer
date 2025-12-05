#include "mysteppingaction.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "G4ios.hh"
#include "G4VProcess.hh"
#include "G4AnalysisManager.hh"
#include <cmath>
#include <cfloat>
#include "G4Exception.hh"
#include "G4RunManager.hh"
#include "runaction.hh"
#include "generator.hh"
#include "G4StepPoint.hh"

MySteppingAction::MySteppingAction() {
}

MySteppingAction::~MySteppingAction() {
}

void MySteppingAction::UserSteppingAction(const G4Step* step) {
    G4Track* track = step->GetTrack();
    auto analysisManager = G4AnalysisManager::Instance();

    // Only count primary protons (trackID == 1)
    if (track->GetDefinition()->GetParticleName() == "proton" && track->GetTrackID() == 1) {
        G4StepPoint* preStepPoint = step->GetPreStepPoint();
        G4StepPoint* postStepPoint = step->GetPostStepPoint();

        if (!preStepPoint || !postStepPoint) return;
        if (!preStepPoint->GetPhysicalVolume()) return;

        G4String preVolume = preStepPoint->GetPhysicalVolume()->GetName();
        G4String postVolume = postStepPoint->GetPhysicalVolume() ? postStepPoint->GetPhysicalVolume()->GetName() : "OutOfWorld";

        // Only count if truly exiting the gold block and NOT entering nickel
        if (preVolume == "physGold" && postVolume != "physGold" &&
            postVolume != "physNickel" &&
            postStepPoint->GetStepStatus() == fGeomBoundary &&
            track->GetTrackStatus() == fAlive) {
            
            // NOTE: IncrementExitCount() was removed from MyRunAction.
            // If you need a run-level counter, increment an accumulable or
            // other mechanism in a thread-safe way inside MyRunAction.

            // Get the kinetic energy of the exiting particle
            G4double energy = track->GetKineticEnergy();

            // Get the incident angle from the particle gun
            auto* primaryGenerator = const_cast<MyPrimaryGenerator*>(
                static_cast<const MyPrimaryGenerator*>(G4RunManager::GetRunManager()->GetUserPrimaryGeneratorAction())
            );
            double incidentAngleRad = primaryGenerator->GetIncidentAngle(); // Incident angle in radians
            double incidentAngleDeg = incidentAngleRad / CLHEP::deg; // Convert to degrees

            // Calculate the rotated incident beam direction
            G4ThreeVector incident = G4ThreeVector(0, std::sin(incidentAngleRad), -std::cos(incidentAngleRad)); // Rotated beam direction
            G4ThreeVector exit = track->GetMomentumDirection(); // Exiting direction (unit vector)

            // Azimuthal acceptance defined relative to the -Z beam axis.
            // For a fixed beam axis = (0,0,-1) the azimuth around that axis can be computed
            // as phi_beam = atan2(-exit.y(), exit.x()). This gives the angle in the plane
            // perpendicular to -Z (consistent sign convention). We then apply the ±0.037° cut.
            const double phi_half_deg = 0.037; // allowed |phi| in degrees (azimuth half-width)
            // compute azimuth around -Z axis (deg)
            double phi_beam = std::atan2(exit.x(), -exit.z());
            double phi_beam_deg = phi_beam / CLHEP::deg;
            if (std::abs(phi_beam_deg) > phi_half_deg) {
                // outside azimuthal acceptance -> do not count
                return;
            }

            // Compute scattered polar relative to the surface normal (+y)
            // Use sign-safe grazing-angle = asin(|dot|) so grazing is always >= 0
            G4ThreeVector surfaceNormal(0.0, 1.0, 0.0);
            G4ThreeVector exitDir = exit.unit();
            G4ThreeVector incidentDir = incident.unit();

            // dot product (may be negative if vector points "below" plane)
            double dot_exit = exitDir.dot(surfaceNormal);
            double dot_inc  = incidentDir.dot(surfaceNormal);

            // clamp to [-1,1] to be safe
            dot_exit = std::clamp(dot_exit, -1.0, 1.0);
            dot_inc  = std::clamp(dot_inc, -1.0, 1.0);

            // grazing angles (angle from surface plane), use absolute to avoid negatives
            double theta_sc_surface_deg = std::asin(std::abs(dot_exit)) / CLHEP::deg;
            double theta_inc_surface_deg = std::asin(std::abs(dot_inc))  / CLHEP::deg;

            // total grazing angle
            double theta_total_surface_deg = theta_sc_surface_deg + theta_inc_surface_deg;

            // New logic: fill H1(1) if theta_total_surface_deg <= 4, else fill H1(2)
            if (theta_total_surface_deg <= 4.5) {
                analysisManager->FillH1(1, theta_total_surface_deg);   // H1(1): small angles
            } else {
                analysisManager->FillH1(2, theta_total_surface_deg);   // H1(2): larger angles
            }
            
            
            // Also fill ntuple for individual event analysis
            // Fill the ntuple (individual event data)
            // G4cout << "energy = " << energy << G4endl;
            // G4cout << "energy diff kev = " << energy / CLHEP::keV  << G4endl;
            analysisManager->FillNtupleDColumn(1, 0, theta_total_surface_deg);
            analysisManager->FillNtupleDColumn(1, 1, energy / CLHEP::keV);
            analysisManager->AddNtupleRow(1);           
        }
    }
}


