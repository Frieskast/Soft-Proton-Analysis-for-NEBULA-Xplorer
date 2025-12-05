#ifndef CONSTRUCTION_MESSENGER_HH
#define CONSTRUCTION_MESSENGER_HH

#include "G4UImessenger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAString.hh"
#include "globals.hh"

class MyPrimaryGenerator;
class MyDetectorConstruction;

class ConstructionMessenger : public G4UImessenger {
public:
    ConstructionMessenger(MyDetectorConstruction* det);
    ~ConstructionMessenger();
    void SetNewValue(G4UIcommand*, G4String) override;
private:
    MyDetectorConstruction* fDet;
    G4UIcmdWithABool* fFilterOnCmd;
    G4UIcmdWithAString* fMirrorTypeCmd;
};

#endif