#include "generatorMessenger.hh"
#include "generator.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

GeneratorMessenger::GeneratorMessenger(MyPrimaryGenerator* gen)
: G4UImessenger(), fGenerator(gen)
{
    fIncidentAngleCmd = new G4UIcmdWithADoubleAndUnit("/gun/incidentAngle", this);
    fIncidentAngleCmd->SetGuidance("Set incident angle of the primary particle.");
    fIncidentAngleCmd->SetParameterName("angle", false);
    fIncidentAngleCmd->SetUnitCategory("Angle");
    fIncidentAngleCmd->SetDefaultValue(1.0);
    fIncidentAngleCmd->SetDefaultUnit("deg");
}

GeneratorMessenger::~GeneratorMessenger() {
    delete fIncidentAngleCmd;
}

void GeneratorMessenger::SetNewValue(G4UIcommand* command, G4String newValue) {
    if (command == fIncidentAngleCmd) {
        fGenerator->SetIncidentAngle(fIncidentAngleCmd->GetNewDoubleValue(newValue));
    }
}