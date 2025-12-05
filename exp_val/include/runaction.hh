#ifndef RUNACTION_HH
#define RUNACTION_HH

#include "G4UserRunAction.hh"
#include "G4Accumulable.hh"
#include "G4SystemOfUnits.hh"
#include "G4String.hh"
#include <string>

// Forward declarations
class G4Run;
class MyPrimaryGenerator;

class MyRunAction : public G4UserRunAction {
  public:
    MyRunAction();
    virtual ~MyRunAction();

    virtual void BeginOfRunAction(const G4Run* run) override;
    virtual void EndOfRunAction(const G4Run* run) override;

    G4String GenerateFileName(const G4Run* run);

    // Keep only the minimal interface used internally
private:

    // run-level parameters (kept in master)
    double fEnergyKeV = 0.0;
    G4double fIncidentAngle = 0.0;
    double fIncidentEnergy = 0.0;
    std::string fPhysicsListName;

    // file/IO and counters
    G4String fPendingFileName;
    G4int fNEvents = 0;

    // accumulable (if you keep using it inside MyRunAction)
    G4Accumulable<G4int> fExitCount;
};
#endif
