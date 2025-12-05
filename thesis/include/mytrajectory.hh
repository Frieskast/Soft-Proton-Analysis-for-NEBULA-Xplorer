#ifndef MYTRAJECTORY_HH
#define MYTRAJECTORY_HH

#include "G4VTrajectory.hh"
#include "G4ThreeVector.hh"
#include "G4ParticleDefinition.hh"
#include "G4Track.hh"
#include <vector>

class MyTrajectory : public G4VTrajectory {
public:
    MyTrajectory(const G4Track* track);
    MyTrajectory(const MyTrajectory& right);
    virtual ~MyTrajectory();

    virtual G4int GetTrackID() const override { return fTrackID; }
    virtual G4int GetParentID() const override { return fParentID; }
    virtual G4String GetParticleName() const override { return fParticleName; }
    virtual G4double GetCharge() const override { return fCharge; }
    virtual G4int GetPDGEncoding() const override { return fPDGCode; }
    virtual G4int GetPointEntries() const override { return static_cast<G4int>(fPoints.size()); }
    virtual G4VTrajectoryPoint* GetPoint(G4int i) const override { return nullptr; } // Not used here
    virtual void ShowTrajectory(std::ostream& os = G4cout) const override {}
    virtual void DrawTrajectory() const override {}
    virtual G4ThreeVector GetInitialMomentum() const override;
    virtual void MergeTrajectory(G4VTrajectory* secondTrajectory) override;

    void AppendStep(const G4Step* step);

    // Access to the stored points
    const std::vector<G4ThreeVector>& GetPoints() const { return fPoints; }

private:
    G4int fTrackID;
    G4int fParentID;
    G4String fParticleName;
    G4double fCharge;
    G4int fPDGCode;
    std::vector<G4ThreeVector> fPoints;
};

#endif