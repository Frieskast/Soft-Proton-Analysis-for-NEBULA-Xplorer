#include "construction.hh"
#include "G4Tubs.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4VisAttributes.hh"
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>

// Redundant optical headers
#include "G4OpticalSurface.hh"       
#include "G4LogicalSkinSurface.hh"   
#include "G4MaterialPropertiesTable.hh" 
#include "G4OpticalPhysics.hh"       

#include "physics.hh"
#include "G4UserLimits.hh"
#include "G4GDMLParser.hh"
#include "runaction.hh"
#include "G4RunManager.hh"

// Constants for the gold/nickel mirror
const double goldThickness = 0.00005 * mm;    // 50 nm
const double nickelThickness = 0.27 * mm;     // 270 um

// Dimensions of the nickel and gold block
const double blockWidth = 0.1 * m;
const double blockHeight = 0.1 * m;

// Dimensions of the detector (REDUNCANT)
const double detectorThickness = 0.01 * m;  // Thickness of the detector
const double detectorRadius = 0.05 * m;     // Radius of the detector
const double detectorXPosition = 0.8 * m;   // Position of the detector along the z-axis

// Previous approach used a swiffeling gold/nickel plate which is now reduncant
MyDetectorConstruction::MyDetectorConstruction() : G4VUserDetectorConstruction() {
    fMessenger = new G4GenericMessenger(this, "/mirror/", "Mirror control");
    fMessenger->DeclareProperty("psi_deg", psi_deg, "Set the mirror grazing angle in degrees");
}

MyDetectorConstruction::~MyDetectorConstruction() {

}

G4VPhysicalVolume* MyDetectorConstruction::Construct() {
    G4NistManager* nist = G4NistManager::Instance();

    // Define materials
    G4Material* gold = nist->FindOrBuildMaterial("G4_Au");
    G4Material* nickel = nist->FindOrBuildMaterial("G4_Ni");
    G4Material* worldMat = nist->FindOrBuildMaterial("G4_Galactic");

    // Define world volume
    G4double xWorld = 1 * m;
    G4double yWorld = 1 * m;
    G4double zWorld = 1 * m;

    G4Box* solidWorld = new G4Box("solidWorld", xWorld, yWorld, zWorld);
    G4LogicalVolume* logicWorld = new G4LogicalVolume(solidWorld, worldMat, "logicWorld");
    G4VPhysicalVolume* physWorld = new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), logicWorld, "physWorld", 0, false, 0, true);

    // Add a step limiter to the vacuum (world volume)
    G4double maxStepVacuum = 0.1 * mm; // Set a maximum step size for the vacuum
    logicWorld->SetUserLimits(new G4UserLimits(maxStepVacuum));

    // Place gold layer (centered at origin)
    G4Box* solidGold = new G4Box("solidGold", blockWidth / 2, goldThickness / 2, blockHeight / 2);
    G4LogicalVolume* logicGold = new G4LogicalVolume(solidGold, gold, "logicGold");
    logicGold->SetUserLimits(new G4UserLimits(0.001 * mm));

    // Set gold color (yellow/gold)
    auto* goldVis = new G4VisAttributes(G4Colour(1.0, 0.84, 0.0)); // RGB for gold
    goldVis->SetForceSolid(true);
    logicGold->SetVisAttributes(goldVis);

    // Place the gold
    new G4PVPlacement(
        0, 
        G4ThreeVector(0., -(goldThickness) / 2, 0.), // Centered at origin
        logicGold,
        "physGold",
        logicWorld,
        false,
        0,
        true
    );

    // Place nickel layer (directly behind gold, in +y)
    G4Box* solidNickel = new G4Box("solidNickel", blockWidth / 2, nickelThickness / 2, blockHeight / 2);
    G4LogicalVolume* logicNickel = new G4LogicalVolume(solidNickel, nickel, "logicNickel");
    logicNickel->SetUserLimits(new G4UserLimits(0.001 * mm));
    
    // Set nickel color (grayish)
    auto* nickelVis = new G4VisAttributes(G4Colour(0.7, 0.7, 0.7)); // RGB for nickel
    nickelVis->SetForceSolid(true);
    logicNickel->SetVisAttributes(nickelVis);

    new G4PVPlacement(
        0,
        G4ThreeVector(0., -(goldThickness + nickelThickness) / 2, 0.), // Flush with gold
        logicNickel,
        "physNickel",
        logicWorld,
        false,
        0,
        true
    );

    return physWorld;
}

void MyDetectorConstruction::ConstructSDandField()
{
    
}
