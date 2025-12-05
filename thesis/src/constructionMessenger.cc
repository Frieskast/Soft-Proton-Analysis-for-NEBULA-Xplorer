#include "constructionMessenger.hh"
#include "construction.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithAString.hh"

ConstructionMessenger::ConstructionMessenger(MyDetectorConstruction* det)
: G4UImessenger(), fDet(det)
{
    fFilterOnCmd = new G4UIcmdWithABool("/det/filterOn", this);
    fFilterOnCmd->SetGuidance("Enable or disable the thermal filter.");
    fFilterOnCmd->SetParameterName("filterOn", false);
    fFilterOnCmd->SetDefaultValue(true);

    fMirrorTypeCmd = new G4UIcmdWithAString("/det/mirrorType", this);
    fMirrorTypeCmd->SetGuidance("Set mirror type: SP, DPH, DCC, VP");
    fMirrorTypeCmd->SetParameterName("type", false);
    fMirrorTypeCmd->SetCandidates("SP DPH DCC VP");
    fMirrorTypeCmd->SetDefaultValue("DPH");
}

ConstructionMessenger::~ConstructionMessenger() {
    delete fFilterOnCmd;
    delete fMirrorTypeCmd;
}

void ConstructionMessenger::SetNewValue(G4UIcommand* cmd, G4String val) {
    if (cmd == fFilterOnCmd) {
        fDet->SetThermalFilterOn(fFilterOnCmd->GetNewBoolValue(val));
    }
    if (cmd == fMirrorTypeCmd) {
        if (val == "SP")
            fDet->SetMirrorType(SINGLE_PARABOLOID);
        else if (val == "DPH")
            fDet->SetMirrorType(DOUBLE_PARA_HYPERBOLIC);
        else if (val == "DCC")
            fDet->SetMirrorType(DOUBLE_CONIC_CONIC);
        else
            G4Exception("ConstructionMessenger::SetNewValue", "InvalidMirrorType", JustWarning, "Unknown mirror type string!");
    }
}