#include "physics.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4EmLivermorePhysics.hh"
#include "G4CoulombScattering.hh"
#include "G4HadronPhysicsQGSP_BIC_HP.hh"
#include "G4HadronElasticPhysicsHP.hh"
#include "G4hIonisation.hh"
#include "G4Proton.hh"
#include "G4ParticleTable.hh"
#include "G4EmParameters.hh"
#include "G4ProductionCutsTable.hh"
#include "G4ProcessManager.hh"
#include "G4DecayPhysics.hh"
#include "G4GoudsmitSaundersonMscModel.hh"
#include "G4UrbanMscModel.hh"
#include "G4hMultipleScattering.hh"
#include "G4UImanager.hh"
#include "G4UIcommand.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4EmStandardPhysicsSS.hh"
#include "G4RunManager.hh"
#include "G4UIcmdWithAString.hh"
#include <string>

// file-scope command pointer and selection storage
static G4UIcmdWithAString* physCmd = nullptr;

// file-scope selection storage: default empty -> not forced too-early
static std::string selectedPhysics = "";  

// allow main to set physics selection before MyPhysicsList is constructed
void SetSelectedPhysics(const std::string& phys) {
    if (phys == "ss" || phys == "option4") {
        selectedPhysics = phys;
        G4cout << "SetSelectedPhysics: " << selectedPhysics << G4endl;
    } else {
        G4cerr << "SetSelectedPhysics: invalid token '" << phys
               << "'. Allowed: option4, ss. Keeping '" << (selectedPhysics.empty() ? "unset" : selectedPhysics) << "'." << G4endl;
    }
}

// new accessor
std::string GetSelectedPhysics() {
    return selectedPhysics.empty() ? std::string("option4") : selectedPhysics;
}

MyPhysicsMessenger::MyPhysicsMessenger() {
    thetaLimitCmd = new G4UIcmdWithADoubleAndUnit("/msc/thetaLimit", this);
    thetaLimitCmd->SetGuidance("Set the MscThetaLimit parameter (in radians)");
    thetaLimitCmd->SetParameterName("thetaLimit", false);
    thetaLimitCmd->SetDefaultUnit("rad");
    thetaLimitCmd->SetRange("thetaLimit>=0.0");

    // create file-scope command so SetNewValue can compare pointer
    physCmd = new G4UIcmdWithAString("/physics/select", this);
    physCmd->SetGuidance("Select physics list (must be called before /run/initialize).");
    physCmd->SetParameterName("phys", false);
    physCmd->SetCandidates("option4 ss"); // only option4 and ss allowed
    physCmd->SetDefaultValue("option4");
}

MyPhysicsMessenger::~MyPhysicsMessenger() {
    delete thetaLimitCmd;
    if (physCmd) { delete physCmd; physCmd = nullptr; }
}

void MyPhysicsMessenger::SetNewValue(G4UIcommand* command, G4String newValue) {
    if (command == thetaLimitCmd) {
        G4double val = thetaLimitCmd->GetNewDoubleValue(newValue);
        // Call the global instance to set the value
        G4EmParameters::Instance()->SetMscThetaLimit(val);
        G4cout << "Set MscThetaLimit to " << val << G4endl;
    } else if (command == physCmd) {
        // store chosen physics; the MyPhysicsList constructor will read this and register appropriate EM physics
        selectedPhysics = std::string(newValue);
        G4cout << "Selected physics: " << selectedPhysics << " (will be used when MyPhysicsList is constructed / on next run manager initialization)" << G4endl;
    }
}

MyPhysicsList::MyPhysicsList() {
    // Create the messenger for physics commands
    new MyPhysicsMessenger();

    // Choose EM physics according to selectedPhysics (default to option4 if still unset)
    if (selectedPhysics == "ss") {
        RegisterPhysics(new G4EmStandardPhysicsSS());
        G4cout << "MyPhysicsList: registered EM = G4EmStandardPhysicsSS" << G4endl;
    } else if (selectedPhysics == "option4") {
        RegisterPhysics(new G4EmStandardPhysics_option4());
        G4cout << "MyPhysicsList: registered EM = G4EmStandardPhysics_option4" << G4endl;
    } else {
        // nothing selected before construction -> fallback and inform user
        RegisterPhysics(new G4EmStandardPhysics_option4());
        G4cout << "MyPhysicsList: no physics selected before construction; defaulting to G4EmStandardPhysics_option4" << G4endl;
    }
    
    // Configure step-cut parameters
    G4EmParameters* emParams = G4EmParameters::Instance();
    emParams->SetStepFunction(0.05, 1 * CLHEP::um);
    // emParams->SetMscThetaLimit(0.05); // Default value for theta limit

    // Set global energy range for production cuts
    G4ProductionCutsTable::GetProductionCutsTable()->SetEnergyRange(0.000001 * CLHEP::eV, 100 * CLHEP::GeV);

    mscMessenger = new MyPhysicsMessenger();
}

MyPhysicsList::~MyPhysicsList() {
    delete mscMessenger;
}

void MyPhysicsList::SetCuts() {
    // Call the base class method
    G4VUserPhysicsList::SetCuts();

    // Set specific cuts for individual particles
    SetCutValue(0.1 * CLHEP::um, "gamma");  // produce low-energy gammas if you need them
    SetCutValue(0.1 * CLHEP::um, "e-");
    SetCutValue(0.1 * CLHEP::um, "e+");
    SetCutValue(1.0 * CLHEP::nm, "proton"); 

    // Print the cuts for verification
    // G4ProductionCutsTable::GetProductionCutsTable()->DumpCouples();
}

void MyPhysicsList::ConstructProcess() {
    // Call the base class implementation to register processes from physics lists
    G4VModularPhysicsList::ConstructProcess();

    // // --- Set Multiple Scattering Model for Protons ---
    // G4ParticleDefinition* proton = G4Proton::ProtonDefinition();
    // G4ProcessManager* pmanager = proton->GetProcessManager();

    // // Remove existing msc processes for protons
    // G4int nProc = pmanager->GetProcessListLength();
    // for (G4int i = nProc - 1; i >= 0; --i) {
    //     G4VProcess* proc = (*pmanager->GetProcessList())[i];
    //     if (proc->GetProcessName() == "msc") {
    //         pmanager->RemoveProcess(proc);
    //     }
    // }

    // // --- Add Single Scattering (Coulomb) ---
    // auto ss = new G4CoulombScattering();
    // pmanager->AddProcess(ss, -1, -1, 1); // Correct ordering for single scattering
}