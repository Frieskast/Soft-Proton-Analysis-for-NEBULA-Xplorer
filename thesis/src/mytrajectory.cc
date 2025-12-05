#include "mytrajectory.hh"
#include "G4Track.hh"
#include "G4Step.hh"

MyTrajectory::MyTrajectory(const G4Track* track)
    : fTrackID(track->GetTrackID()),
      fParentID(track->GetParentID()),
      fParticleName(track->GetDefinition()->GetParticleName()),
      fCharge(track->GetDefinition()->GetPDGCharge()),
      fPDGCode(track->GetDefinition()->GetPDGEncoding())
{
    fPoints.push_back(track->GetPosition());
}

MyTrajectory::MyTrajectory(const MyTrajectory& right)
    : G4VTrajectory(right),
      fTrackID(right.fTrackID),
      fParentID(right.fParentID),
      fParticleName(right.fParticleName),
      fCharge(right.fCharge),
      fPDGCode(right.fPDGCode),
      fPoints(right.fPoints)
{}

MyTrajectory::~MyTrajectory() {}

void MyTrajectory::AppendStep(const G4Step* step) {
    fPoints.push_back(step->GetPostStepPoint()->GetPosition());
}

// Return a default value or store the initial momentum in your constructor if you want
G4ThreeVector MyTrajectory::GetInitialMomentum() const {
    return G4ThreeVector(); // or store/return the actual initial momentum
}

void MyTrajectory::MergeTrajectory(G4VTrajectory* /*secondTrajectory*/) {
    
}