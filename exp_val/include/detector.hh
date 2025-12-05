#ifndef DETECTOR_HH
#define DETECTOR_HH

#include "G4VSensitiveDetector.hh"
// #include "G4VModularPhysicsList.hh"
// #include "G4EmStandardPhysics.hh"
// #include "G4OpticalPhysics.hh"
// #include "G4RadioactiveDecayPhysics.hh"
// #include "G4DecayPhysics.hh"

class MySensitiveDetector : public G4VSensitiveDetector
{
public:
    MySensitiveDetector(G4String);
    ~MySensitiveDetector();

private:
    virtual G4bool ProcessHits(G4Step *, G4TouchableHistory *);
};

#endif
