#ifndef GENERATORMESSENGER_HH
#define GENERATORMESSENGER_HH

#include "G4UImessenger.hh"
class G4UIcmdWithADoubleAndUnit;
class MyPrimaryGenerator;

class GeneratorMessenger : public G4UImessenger {
public:
    GeneratorMessenger(MyPrimaryGenerator* gen);
    virtual ~GeneratorMessenger();

    virtual void SetNewValue(G4UIcommand* command, G4String newValue) override;

private:
    MyPrimaryGenerator* fPrimaryGenerator;
    G4UIcmdWithADoubleAndUnit* fIncidentAngleCmd;
    G4UIcmdWithADoubleAndUnit* fEnergyCmd;
};

#endif