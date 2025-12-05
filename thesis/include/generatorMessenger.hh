#ifndef GENERATOR_MESSENGER_HH
#define GENERATOR_MESSENGER_HH

#include "G4UImessenger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "globals.hh"

class MyPrimaryGenerator;

class GeneratorMessenger : public G4UImessenger {
public:
    GeneratorMessenger(MyPrimaryGenerator* gen);
    ~GeneratorMessenger();

    void SetNewValue(G4UIcommand* command, G4String newValue) override;

private:
    MyPrimaryGenerator* fGenerator;
    G4UIcmdWithADoubleAndUnit* fIncidentAngleCmd;
};

#endif