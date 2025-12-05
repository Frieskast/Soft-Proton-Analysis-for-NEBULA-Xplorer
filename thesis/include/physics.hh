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
#include "G4hPairProduction.hh"
#include "G4hBremsstrahlung.hh"
#include "G4hIonisation.hh"
#include "G4hMultipleScattering.hh"
#include "G4UImessenger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include <string>

// Allow main to set physics selection before MyPhysicsList is constructed
void SetSelectedPhysics(const std::string& phys);
// return currently selected physics (fallback "option4" if unset)
std::string GetSelectedPhysics();

class MyPhysicsMessenger : public G4UImessenger {
public:
    MyPhysicsMessenger();
    ~MyPhysicsMessenger() override;
    void SetNewValue(G4UIcommand*, G4String) override;

    G4UIcmdWithADoubleAndUnit* thetaLimitCmd = nullptr;
};

class MyPhysicsList : public G4VModularPhysicsList
{
public:
    MyPhysicsList();
    ~MyPhysicsList();

    void SetCuts() override;
    void ConstructProcess() override; // Add this declaration
    void ConstructEM();       // Electromagnetic processes
    void ConstructGeneral();  // General processes
    void SetMscThetaLimit(G4double val);
private:
    MyPhysicsMessenger* mscMessenger = nullptr;
    void SetupMessenger();
};

#endif
