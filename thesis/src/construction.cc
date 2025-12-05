#include "construction.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4Polycone.hh"
#include "G4RotationMatrix.hh"
#include "G4Tubs.hh"
#include "G4SubtractionSolid.hh"
#include <utility>
#include "G4VisAttributes.hh"
#include "G4SDManager.hh"
#include "detector.hh"
#include "G4LogicalVolumeStore.hh"
#include "mirror_single_paraboloid.hh"
#include "mirror_wolter1.hh"
#include "mirror_double_conic.hh"
#include "constructionMessenger.hh"
#include "G4UserLimits.hh"

double FindParaboloidPForYTip(double y_tip_target, double shellLength, double focalLength, double rMax, double x_start);

MyDetectorConstruction::MyDetectorConstruction()
 : G4VUserDetectorConstruction()
{
    fMessenger = new ConstructionMessenger(this);
}

MyDetectorConstruction::~MyDetectorConstruction() {}

G4VPhysicalVolume* MyDetectorConstruction::Construct() {
    // Materials
    G4NistManager* nist = G4NistManager::Instance();
    G4Material* gold = nist->FindOrBuildMaterial("G4_Au");
    G4Material* aluminium = nist->FindOrBuildMaterial("G4_Al");
    G4Material* worldMat = nist->FindOrBuildMaterial("G4_Galactic");
    G4Material* epoxy = new G4Material("Epoxy", 1.18 * g/cm3, 3);
    epoxy->AddElement(nist->FindOrBuildElement("C"), 18);
    epoxy->AddElement(nist->FindOrBuildElement("H"), 19);
    epoxy->AddElement(nist->FindOrBuildElement("O"), 3);

    // World
    G4double worldSize_x = 1.2 * m; 
    G4double worldSize_y = 0.2 * m;
    G4double worldSize_z = 0.2 * m;

    G4Box* solidWorld = new G4Box("World", worldSize_x, worldSize_y, worldSize_z);
    G4LogicalVolume* logicWorld = new G4LogicalVolume(solidWorld, worldMat, "WorldLV");
    G4VPhysicalVolume* physWorld = new G4PVPlacement(
        0, G4ThreeVector(0, 0, 0), logicWorld, "World", 0, false, 0, true);

    // Make the world visible as a wireframe outline (light blue)
    G4VisAttributes* worldVis = new G4VisAttributes(G4Colour(0.2, 0.6, 1.0));
    worldVis->SetForceWireframe(true);
    logicWorld->SetVisAttributes(worldVis);

    // Choose which mirror to build
    MirrorParams params(
        fMirrorType,       // Set type from member variable
        0.8 * m,           // focalLength
        0.2 * m,           // length
        0.009 * m,         // rMin
        0.047 * m          // rMax
    );

    // This line is now redundant as the type is set during construction
    // params.type = fMirrorType;

    // Declare bar parameters before the switch
    double bar_x_start = 0.0;
    double bar_x_end = 0.0;
    double bar_half_length = 0.0;
    double bar_center_x = 0.0;

    // Call the appropriate builder
    switch (params.type) {
        case SINGLE_PARABOLOID:
        BuildNestedParaboloids(params, logicWorld, gold, epoxy, aluminium);
            {
                double bar_radius = 9.0 * mm;
                double bar_hole_radius = 0.0 * mm; // Change this to include or exclude the opening in dumbbell
                bar_x_start = 0.75 * m;
                bar_x_end = 0.85 * m;
                bar_half_length = (bar_x_end - bar_x_start) / 2.0;
                bar_center_x = (bar_x_start + bar_x_end) / 2.0;
                
                G4Tubs* solidBar = new G4Tubs("AlBar", bar_hole_radius, bar_radius, bar_half_length, 0, 2*M_PI);
                G4LogicalVolume* logicBar = new G4LogicalVolume(solidBar, aluminium, "AlBarLV");
                G4RotationMatrix* rotY90 = new G4RotationMatrix();
                rotY90->rotateY(90.*deg);
                new G4PVPlacement(rotY90, G4ThreeVector(bar_center_x, 0, 0), logicBar, "AlBarPV", logicWorld, false, 0, true);

                G4VisAttributes* barVis = new G4VisAttributes(G4Colour(0.2, 0.2, 0.2)); // dark grey
                barVis->SetForceSolid(true);
                logicBar->SetVisAttributes(barVis);

                // --- Add 3mm thick wall around the concentrators ---
                double wall_inner_radius = params.rMax;
                double wall_outer_radius = params.rMax + 3.0 * mm;
                double wall_half_length = bar_half_length;
                double wall_center_x = bar_center_x;

                G4Tubs* solidWall = new G4Tubs("Wall", wall_inner_radius, wall_outer_radius, wall_half_length, 0, 2*M_PI);
                G4LogicalVolume* logicWall = new G4LogicalVolume(solidWall, aluminium, "WallLV");
                G4RotationMatrix* rotY90Wall = new G4RotationMatrix();
                rotY90Wall->rotateY(90.*deg);
                new G4PVPlacement(rotY90Wall, G4ThreeVector(wall_center_x, 0, 0), logicWall, "WallPV", logicWorld, false, 0, true);

                G4VisAttributes* wallVis = new G4VisAttributes(G4Colour(0.2, 0.2, 0.2, 0.2)); // dark grey
                wallVis->SetForceSolid(true);
                logicWall->SetVisAttributes(wallVis);
            }
            break;

        case DOUBLE_PARA_HYPERBOLIC:
            BuildNestedWolterI(params, logicWorld, gold, epoxy, aluminium);
            {
                double bar_radius = 9.0 * mm;
                double bar_hole_radius = 0.0 * mm; // Change this to include or exclude the opening in dumbbell
                bar_x_start = 0.7 * m;
                bar_x_end = 0.9 * m;
                bar_half_length = (bar_x_end - bar_x_start) / 2.0;
                bar_center_x = (bar_x_start + bar_x_end) / 2.0;

                G4Tubs* solidBar = new G4Tubs("AlBar", bar_hole_radius, bar_radius, bar_half_length, 0, 2*M_PI);
                G4LogicalVolume* logicBar = new G4LogicalVolume(solidBar, aluminium, "AlBarLV");
                G4RotationMatrix* rotY90 = new G4RotationMatrix();
                rotY90->rotateY(90.*deg);
                new G4PVPlacement(rotY90, G4ThreeVector(bar_center_x, 0, 0), logicBar, "AlBarPV", logicWorld, false, 0, true);

                G4VisAttributes* barVis = new G4VisAttributes(G4Colour(0.2, 0.2, 0.2)); // dark grey
                barVis->SetForceSolid(true);
                logicBar->SetVisAttributes(barVis);

                // --- Add 3mm thick wall around the concentrators ---
                double wall_inner_radius = params.rMax;
                double wall_outer_radius = params.rMax + 3.0 * mm;
                double wall_half_length = bar_half_length;
                double wall_center_x = bar_center_x;

                G4Tubs* solidWall = new G4Tubs("Wall", wall_inner_radius, wall_outer_radius, wall_half_length, 0, 2*M_PI);
                G4LogicalVolume* logicWall = new G4LogicalVolume(solidWall, aluminium, "WallLV");
                G4RotationMatrix* rotY90Wall = new G4RotationMatrix();
                rotY90Wall->rotateY(90.*deg);
                new G4PVPlacement(rotY90Wall, G4ThreeVector(wall_center_x, 0, 0), logicWall, "WallPV", logicWorld, false, 0, true);

                G4VisAttributes* wallVis = new G4VisAttributes(G4Colour(0.2, 0.2, 0.2, 0.2)); // dark grey
                wallVis->SetForceSolid(true);
                logicWall->SetVisAttributes(wallVis);
            }
            break;

        case DOUBLE_CONIC_CONIC:
            BuildDoubleConic(params, logicWorld, gold, epoxy, aluminium);
            {
                double bar_radius = 9.0 * mm;
                double bar_hole_radius = 0.0 * mm; // Change this to include or exclude the opening in dumbbell
                bar_x_start = 0.7 * m;
                bar_x_end = 0.9 * m;
                bar_half_length = (bar_x_end - bar_x_start) / 2.0;
                bar_center_x = (bar_x_start + bar_x_end) / 2.0;

                G4Tubs* solidBar = new G4Tubs("AlBar", bar_hole_radius, bar_radius, bar_half_length, 0, 2*M_PI);
                G4LogicalVolume* logicBar = new G4LogicalVolume(solidBar, aluminium, "AlBarLV");
                G4RotationMatrix* rotY90 = new G4RotationMatrix();
                rotY90->rotateY(90.*deg);
                new G4PVPlacement(rotY90, G4ThreeVector(bar_center_x, 0, 0), logicBar, "AlBarPV", logicWorld, false, 0, true);

                G4VisAttributes* barVis = new G4VisAttributes(G4Colour(0.2, 0.2, 0.2)); // dark grey
                barVis->SetForceSolid(true);
                logicBar->SetVisAttributes(barVis);

                // --- Add 3mm thick wall around the concentrators ---
                double wall_inner_radius = params.rMax;
                double wall_outer_radius = params.rMax + 3.0 * mm;
                double wall_half_length = bar_half_length;
                double wall_center_x = bar_center_x;

                G4Tubs* solidWall = new G4Tubs("Wall", wall_inner_radius, wall_outer_radius, wall_half_length, 0, 2*M_PI);
                G4LogicalVolume* logicWall = new G4LogicalVolume(solidWall, aluminium, "WallLV");
                G4RotationMatrix* rotY90Wall = new G4RotationMatrix();
                rotY90Wall->rotateY(90.*deg);
                new G4PVPlacement(rotY90Wall, G4ThreeVector(wall_center_x, 0, 0), logicWall, "WallPV", logicWorld, false, 0, true);

                G4VisAttributes* wallVis = new G4VisAttributes(G4Colour(0.2, 0.2, 0.2, 0.2)); // dark grey
                wallVis->SetForceSolid(true);
                logicWall->SetVisAttributes(wallVis);
            }
            break;
    }


    // The Silicon Detector will be centered at -250um, so it spans from 0 to -500um.
    G4double si_detector_center_x = -250.0 * um;
    // The Aluminum Focal Plate will be centered at -750um, so it spans from -500um to -1000um.
    G4double focal_plate_center_x = -750.0 * um;

    // Add detector volume (Aluminum Focal Plate)
    double focal_half_size = 0.2 * m;
    double focal_thickness = 0.5 * mm;

    // G4Box is along z by default, so rotate to y-z plane
    G4Box* solidDet = new G4Box("Detector", focal_half_size, focal_half_size, focal_thickness/2);
    G4LogicalVolume* logicDet = new G4LogicalVolume(solidDet, aluminium, "DetectorLV");

    // Rotate so normal is along x-axis
    G4RotationMatrix* rotY90det = new G4RotationMatrix();
    rotY90det->rotateY(90.*deg);

    // Place the focal plate at its center ---
    new G4PVPlacement(rotY90det, G4ThreeVector(focal_plate_center_x, 0, 0), logicDet, "DetectorPV", logicWorld, false, 0, true);

    // Visualization
    G4VisAttributes* detVis = new G4VisAttributes(G4Colour(0.75, 0.75, 0.75, 1)); // silver
    detVis->SetForceSolid(true);
    logicDet->SetVisAttributes(detVis);

    // --- Add stop plate with 5cm hole at correct x position ---
    double stop_plate_thickness = 1.0 * mm;
    double plate_size_y = worldSize_y; // 0.2 m
    double plate_size_z = worldSize_z; // 0.2 m

    // Choose plate x position based on mirror type
    double plate_center_x = 0.85 * m; // default for paraboloid
    if (params.type == DOUBLE_PARA_HYPERBOLIC || params.type == DOUBLE_CONIC_CONIC) {
        plate_center_x = 0.90 * m;
    }

    // Create the box (plate) in the y/z plane at x = plate_center_x
    G4Box* solidPlateBox = new G4Box("PlateBox", stop_plate_thickness/2, plate_size_y, plate_size_z);

    // Create the cylindrical hole (axis along x, so rotate by 90 deg around y)
    G4Tubs* solidPlateHole = new G4Tubs("PlateHole", 0, 0.05 * m, plate_size_z + 1*cm, 0, 2*M_PI);
    G4RotationMatrix* rotY90 = new G4RotationMatrix();
    rotY90->rotateY(90.*deg);

    // Subtract the hole from the plate
    G4SubtractionSolid* solidPlate = new G4SubtractionSolid(
        "StopPlate", solidPlateBox,
        solidPlateHole,
        rotY90, // rotate the hole so it's along x
        G4ThreeVector(0, 0, 0)
    );

    G4LogicalVolume* logicStopPlate = new G4LogicalVolume(solidPlate, aluminium, "StopPlateLV");

    // Place the plate at the correct x, no rotation needed
    new G4PVPlacement(
        0, // no rotation
        G4ThreeVector(plate_center_x, 0, 0),
        logicStopPlate, "StopPlatePV", logicWorld, false, 0, true
    );

    // Visualization: silver, semi-transparent
    G4VisAttributes* stopPlateVis = new G4VisAttributes(G4Colour(0.75, 0.75, 0.75, 0.5)); // silver
    stopPlateVis->SetForceSolid(true);
    logicStopPlate->SetVisAttributes(stopPlateVis);

    // --- Add thermal filter in front of the concentrator ---
    if (fThermalFilterOn) {
        double filter_center_x = 0.85 * m; // default for single paraboloid
        if (params.type == DOUBLE_PARA_HYPERBOLIC || params.type == DOUBLE_CONIC_CONIC) {
            filter_center_x = 0.90 * m;
        }
        double filter_radius = params.rMax + 3.0 * mm;
        double polyimide_thick = 0.2 * um; // 200 nm
        double aluminium_thick = 0.03 * um; // 30 nm
        filter_center_x += 0.115 * um; // fixing overlap
        // Rotation so normal is along x-axis
        G4RotationMatrix* rotY90filter = new G4RotationMatrix();
        rotY90filter->rotateY(90.*deg);

        // Polyimide layer (centered at filter_center_x - aluminium_thick/2)
        G4Material* polyimide = nist->FindOrBuildMaterial("G4_KAPTON");
        G4Tubs* solidPoly = new G4Tubs("FilterPoly", 0, filter_radius, polyimide_thick/2, 0, 2*M_PI);
        G4LogicalVolume* logicPoly = new G4LogicalVolume(solidPoly, polyimide, "FilterPolyLV");
        new G4PVPlacement(
            rotY90filter,
            G4ThreeVector(filter_center_x - aluminium_thick/2, 0, 0),
            logicPoly, "FilterPolyPV", logicWorld, false, 0, true);

        // Aluminium layer (centered at filter_center_x + polyimide_thick/2)
        G4Tubs* solidAl = new G4Tubs("FilterAl", 0, filter_radius, aluminium_thick/2, 0, 2*M_PI);
        G4LogicalVolume* logicAl = new G4LogicalVolume(solidAl, aluminium, "FilterAlLV");
        new G4PVPlacement(
            rotY90filter,
            G4ThreeVector(filter_center_x + polyimide_thick/2, 0, 0),
            logicAl, "FilterAlPV", logicWorld, false, 0, true);

        // Visualization
        G4VisAttributes* polyVis = new G4VisAttributes(G4Colour(1.0, 0.7, 0.2, 0.4)); // orange, semi-transparent
        polyVis->SetForceSolid(true);
        logicPoly->SetVisAttributes(polyVis);

        G4VisAttributes* alVis = new G4VisAttributes(G4Colour(0.8, 0.8, 0.8, 0.1)); // silver, semi-transparent
        alVis->SetForceSolid(true);
        logicAl->SetVisAttributes(alVis);
    }

    // silicon detector: mass vs active sensitive area 
    G4double det_mass_radius   = 4.72 * mm;  // physical mass radius
    G4double det_active_radius = 4.00 * mm;  // active/sensitive radius (aperture)
    G4double det_thickness     = 0.5  * mm;  // 500 um
    G4Material* silicon = nist->FindOrBuildMaterial("G4_Si");

    // mass volume (Si mass, not necessarily sensitive) - outer cylinder
    G4Tubs* solidSiMass = new G4Tubs("SiMass", 0, det_mass_radius, det_thickness/2, 0, 2*M_PI);
    G4LogicalVolume* logicSiMass = new G4LogicalVolume(solidSiMass, silicon, "SiMassLV");
    G4RotationMatrix* rotY90Si = new G4RotationMatrix();
    rotY90Si->rotateY(90.*deg);

    // Place the silicon detector at its center 
    new G4PVPlacement(rotY90Si, G4ThreeVector(si_detector_center_x, 0, 0), logicSiMass, "SiMassPV", logicWorld, false, 0, true);

    // active (sensitive) inner volume - placed at same center
    G4Tubs* solidSiActive = new G4Tubs("SiDet", 0, det_active_radius, det_thickness/2, 0, 2*M_PI);
    G4LogicalVolume* logicSiActive = new G4LogicalVolume(solidSiActive, silicon, "SiDetLV");
    new G4PVPlacement(0, G4ThreeVector(0,0,0), logicSiActive, "SiDetInnerPV", logicSiMass, false, 0, true);
    // Note: sensitive detector is attached to "SiDetLV" in ConstructSDandField()

    // Limit step length in the sensitive silicon to 1 micrometer
    G4double maxStepSi = 1.0 * um;                // desired maximum step
    G4UserLimits* ulSi = new G4UserLimits(maxStepSi);
    logicSiActive->SetUserLimits(ulSi);

    // --- 150nm Si3N4 window sized to active aperture ---
    G4double win_thickness = 0.15 * um; // 150 nm

    G4Material* si3n4 = nist->FindOrBuildMaterial("G4_SILICON_NITRIDE");
    if (!si3n4) {
        G4Element* elSi = nist->FindOrBuildElement("Si");
        G4Element* elN  = nist->FindOrBuildElement("N");
        si3n4 = new G4Material("Si3N4_custom", 3.17 * g/cm3, 2);
        si3n4->AddElement(elSi, 3);
        si3n4->AddElement(elN, 4);
    }
    G4Tubs* solidWin = new G4Tubs("Si3N4Win", 0, det_active_radius, win_thickness/2, 0, 2*M_PI);
    G4LogicalVolume* logicWin = new G4LogicalVolume(solidWin, si3n4, "Si3N4WinLV");
    // Place window just in front of silicon active face (which is now at x=0)
    // The window center will be at x = +win_thickness/2
    new G4PVPlacement(rotY90Si, G4ThreeVector(win_thickness/2, 0, 0), logicWin, "Si3N4WinPV", logicWorld, false, 0, true);

    // Visualization attributes
    G4VisAttributes* siMassVis = new G4VisAttributes(G4Colour(0.0, 0.6, 0.0, 0.25)); // outer mass translucent
    siMassVis->SetForceSolid(true);
    logicSiMass->SetVisAttributes(siMassVis);
    G4VisAttributes* siActiveVis = new G4VisAttributes(G4Colour(0.0, 0.8, 0.0, 0.9)); // inner active bright
    siActiveVis->SetForceSolid(true);
    logicSiActive->SetVisAttributes(siActiveVis);
    G4VisAttributes* winVis = new G4VisAttributes(G4Colour(0.0, 0.0, 1.0, 0.5)); // blue, semi-transparent
    winVis->SetForceSolid(true);
    logicWin->SetVisAttributes(winVis);

    G4cout << "Construct called, fMirrorType: " << fMirrorType << G4endl;

    return physWorld;
}

G4String MyDetectorConstruction::GetMirrorTypeString() const {
    switch (fMirrorType) {
        case SINGLE_PARABOLOID:       return "SP";
        case DOUBLE_PARA_HYPERBOLIC:  return "DPH";
        case DOUBLE_CONIC_CONIC:      return "DCC";
        default:                      return "unknown";
    }
}

void MyDetectorConstruction::ConstructSDandField() {
    // --- Create Sensitive Detectors ---
    // The names "SiDet" and "FocalDet" are used to find the detectors later.
    G4SDManager* sdManager = G4SDManager::GetSDMpointer();
    auto* detSD = new MySensitiveDetector("DetectorSD");
    sdManager->AddNewDetector(detSD);

    // Attach to the detector logical volume
    auto* logicDet = G4LogicalVolumeStore::GetInstance()->GetVolume("DetectorLV");
    if (logicDet) {
        logicDet->SetSensitiveDetector(detSD);
    }

    // Attach to the silicon detector logical volume
    auto* logicSiDet = G4LogicalVolumeStore::GetInstance()->GetVolume("SiDetLV");
    if (logicSiDet) {
        logicSiDet->SetSensitiveDetector(detSD);
    }
}

void MyDetectorConstruction::SetMirrorType(MirrorType t) {
    fMirrorType = t;
    G4cout << "SetMirrorType called: " << t << G4endl;
}





