#ifndef PHYSICS_HH
#define PHYSICS_HH

#include "G4VModularPhysicsList.hh"
#include "G4EmStandardPhysics.hh"
#include "G4OpticalPhysics.hh"
#include "G4EmLowEPPhysics.hh"  
#include "G4DecayPhysics.hh"
#include "G4RadioactiveDecayPhysics.hh"
#include "G4HadronPhysicsQGSP_BERT_HP.hh" // Hadronic physics
#include "G4IonPhysics.hh"            // Ion physics
#include "G4HadronElasticPhysicsHP.hh"
#include "G4EmStandardPhysics_option3.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4hPairProduction.hh"
#include "G4hBremsstrahlung.hh"
#include "G4hIonisation.hh"
#include "G4hMultipleScattering.hh"
#include "G4UImessenger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"

class MyPhysicsMessenger : public G4UImessenger {
public:
    MyPhysicsMessenger();
    virtual ~MyPhysicsMessenger();

    virtual void SetNewValue(G4UIcommand* command, G4String newValue);

private:
    G4UIcmdWithADoubleAndUnit* thetaLimitCmd;
    G4UIcmdWithAString* physicsListCmd;  // Add this line
};

class MyPhysicsList : public G4VModularPhysicsList {
public:
    MyPhysicsList();  // Default constructor
    virtual ~MyPhysicsList();

    virtual void SetCuts();
    virtual void ConstructProcess();
};

#endif
