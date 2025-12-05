#ifndef CONSTRUCTION_HH
#define CONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"
#include "G4Types.hh"

class G4VPhysicalVolume;
class ConstructionMessenger;

enum MirrorType {
    SINGLE_PARABOLOID,
    DOUBLE_PARA_HYPERBOLIC,
    DOUBLE_CONIC_CONIC,
    VALIDATION_PLATE
};

struct MirrorParams {
    MirrorType type;
    double focalLength;
    double length;
    double rMin;
    double rMax;

    MirrorParams(MirrorType t, double f, double l, double r_min, double r_max)
        : type(t), focalLength(f), length(l), rMin(r_min), rMax(r_max) {}
};

class MyDetectorConstruction : public G4VUserDetectorConstruction
{
public:
    MyDetectorConstruction();
    ~MyDetectorConstruction();

    virtual G4VPhysicalVolume* Construct();
    virtual void ConstructSDandField();

    void SetMirrorType(MirrorType t);
    MirrorType GetMirrorType() const { return fMirrorType; }
    G4String GetMirrorTypeString() const;

    void SetThermalFilterOn(G4bool val) { fThermalFilterOn = val; }
    G4bool GetThermalFilterOn() const { return fThermalFilterOn; }

private:
    MirrorType fMirrorType = SINGLE_PARABOLOID;
    G4bool fThermalFilterOn = false;
    ConstructionMessenger* fMessenger;
};

#endif
