#ifndef VALIDATIONSTEPPINGACTION_HH
#define VALIDATIONSTEPPINGACTION_HH

#include "G4UserSteppingAction.hh"
#include "G4Step.hh"
#include "G4ThreeVector.hh"

class ValidationSteppingAction : public G4UserSteppingAction
{
public:
    ValidationSteppingAction();
    ~ValidationSteppingAction();

    virtual void UserSteppingAction(const G4Step*);
};

#endif