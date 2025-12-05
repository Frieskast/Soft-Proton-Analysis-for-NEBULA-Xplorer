#include "mytrackingaction.hh"
#include "mytrajectory.hh"
#include "G4TrackingManager.hh"
#include "G4Track.hh"

MyTrackingAction::MyTrackingAction() {}
MyTrackingAction::~MyTrackingAction() {}

void MyTrackingAction::PreUserTrackingAction(const G4Track*) {
    fpTrackingManager->SetStoreTrajectory(true);
}

void MyTrackingAction::PostUserTrackingAction(const G4Track*) {}