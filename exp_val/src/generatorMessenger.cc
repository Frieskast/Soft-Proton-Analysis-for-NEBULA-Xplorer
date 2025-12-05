#include "generatorMessenger.hh"
#include "generator.hh"
#include "runaction.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4RunManager.hh"
#include "G4Threading.hh"
#include "G4SystemOfUnits.hh"

GeneratorMessenger::GeneratorMessenger(MyPrimaryGenerator* gen)
: G4UImessenger(), fPrimaryGenerator(gen)
{
    fIncidentAngleCmd = new G4UIcmdWithADoubleAndUnit("/gun/incidentAngle", this);
    fIncidentAngleCmd->SetGuidance("Set incident angle of the primary particle.");
    fIncidentAngleCmd->SetParameterName("angle", false);
    fIncidentAngleCmd->SetUnitCategory("Angle");
    fIncidentAngleCmd->SetDefaultUnit("deg");

    fEnergyCmd = new G4UIcmdWithADoubleAndUnit("/gun/energy", this);
    fEnergyCmd->SetGuidance("Set energy of the primary particle.");
    fEnergyCmd->SetParameterName("energy", false);
    fEnergyCmd->SetUnitCategory("Energy");
    fEnergyCmd->SetDefaultUnit("keV");
}

GeneratorMessenger::~GeneratorMessenger() {
    delete fIncidentAngleCmd;
    delete fEnergyCmd;
}

void GeneratorMessenger::SetNewValue(G4UIcommand* command, G4String newValue) {
    // Incident angle: GetNewDoubleValue returns internal units (radian)
    if (command == fIncidentAngleCmd) {
        G4double angleRad = fIncidentAngleCmd->GetNewDoubleValue(newValue);
        if (fPrimaryGenerator) fPrimaryGenerator->SetIncidentAngle(angleRad);
        return;
    }

    // Energy: GetNewDoubleValue returns internal units (e.g. keV*CLHEP::keV)
    if (command == fEnergyCmd) {
        G4double energyInternal = fEnergyCmd->GetNewDoubleValue(newValue);
        if (fPrimaryGenerator) fPrimaryGenerator->SetEnergy(energyInternal);
        return;
    }
}