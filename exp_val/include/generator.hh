#ifndef GENERATOR_HH
#define GENERATOR_HH

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "Randomize.hh"

// Forward declaration
class GeneratorMessenger;

class MyPrimaryGenerator : public G4VUserPrimaryGeneratorAction
{
public:
    MyPrimaryGenerator();
    ~MyPrimaryGenerator();

    virtual void GeneratePrimaries(G4Event* anEvent);

    // Set/get incident angle: now stored in generator as radians
    // Provide and document that SetIncidentAngle/GetIncidentAngle use radians
    void SetIncidentAngle(G4double angleRad) {
        fIncidentAngle = angleRad; // store radians
    }
    G4double GetIncidentAngle() const { return fIncidentAngle; } // returns radians

    void SetEnergy(G4double energy); // expects internal units (e.g. keV*CLHEP::keV)
    // Return the configured particle energy in keV (convenience for run-action)
    G4double GetEnergyKeV() const { return fParticleGun->GetParticleEnergy() / CLHEP::keV; }

private:
    G4ParticleGun* fParticleGun;
    GeneratorMessenger* fMessenger;
    G4double fIncidentAngle;
};

#endif