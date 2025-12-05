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
#include "G4OpBoundaryProcess.hh"

MySteppingAction::MySteppingAction(){
    
}

MySteppingAction::~MySteppingAction() {
    
}

void MySteppingAction::UserSteppingAction(const G4Step* step) {
    G4Track* track = step->GetTrack();
    G4int trackID = track->GetTrackID();

    // Get pre and post step points
    G4StepPoint* preStepPoint = step->GetPreStepPoint();
    G4StepPoint* postStepPoint = step->GetPostStepPoint();

    if (track->GetDefinition()->GetParticleName() == "proton") {
        // Store trajectory
        G4ThreeVector pos = preStepPoint->GetPosition();
        trackTrajectories[trackID].push_back(pos);
    }

    // A reflection is defined as the FIRST time a proton enters a mirror volume.
    if (preStepPoint->GetStepStatus() == fGeomBoundary) {
        if (track->GetDefinition()->GetParticleName() != "proton") {
        return;
    }

        G4LogicalVolume* lv = preStepPoint->GetTouchableHandle()->GetVolume()->GetLogicalVolume();
        G4String lvName = lv->GetName();
        
        // --- MODIFICATION: Check for any mirror component ---
        bool isGold = (lvName.find("Gold") != std::string::npos);
        bool isEpoxy = (lvName.find("Epoxy") != std::string::npos);
        bool isAluminium = (lvName.find("Aluminium") != std::string::npos);

        if (isGold || isEpoxy || isAluminium) {
            // Only record the first time the track enters this specific volume
            if (trackEnteredMirrors[trackID].find(lvName) == trackEnteredMirrors[trackID].end()) {
                trackEnteredMirrors[trackID].insert(lvName);
                ReflectionHit hit;
                hit.position = preStepPoint->GetPosition();
                hit.materialName = lv->GetMaterial()->GetName();
                hit.volumeName = lvName;
                trackMirrorHits[trackID].push_back(hit); // Add to the unified vector

                // // --- DIAGNOSTIC PRINT ---
                // G4cout << "STEPPING_DEBUG (Track " << trackID << "): Mirror assembly hit in '" 
                //        << lvName << "' (Material: " << hit.materialName << ")" << G4endl;
            }
        }
    }
}