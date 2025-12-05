#ifndef CONSTRUCTION_HH
#define CONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Box.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4LogicalVolume.hh"
#include "G4GenericMessenger.hh"
#include "G4SystemOfUnits.hh"

#include "detector.hh"


class MyDetectorConstruction : public G4VUserDetectorConstruction
{
public:
	MyDetectorConstruction();
	~MyDetectorConstruction();

	//G4LogicalVolume* GetScoringVolume() const { return fScoringVolume; }

	virtual G4VPhysicalVolume* Construct();

private:
    G4LogicalVolume *logicDetector;
    double psi_deg = 45; // default value
    G4GenericMessenger* fMessenger = nullptr;
    virtual void ConstructSDandField();
};
#endif // !CONSTRUCTION_HH
