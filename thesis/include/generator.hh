#ifndef GENERATOR_HH
#define GENERATOR_HH

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4IonTable.hh"
#include "G4Geantino.hh"
#include "G4GeneralParticleSource.hh"

// Forward declaration
class GeneratorMessenger;

class MyPrimaryGenerator : public G4VUserPrimaryGeneratorAction
{
public:
    MyPrimaryGenerator();
    ~MyPrimaryGenerator();

    virtual void GeneratePrimaries(G4Event*);

private:
    G4double fIncidentAngle = 1.0 * deg; // Default value

public:
    void SetIncidentAngle(G4double angle) { fIncidentAngle = angle; }
    G4double GetIncidentAngle() const { return fIncidentAngle; }
    void SetParticleType(const G4String& name);

    G4ParticleGun* fParticleGun;
    G4GeneralParticleSource* fGPS;
    GeneratorMessenger* fMessenger;

    G4ParticleGun* GetParticleGun() const { return fParticleGun; }
};

#endif