#include "physics.hh"
#include "G4EmStandardPhysics_option3.hh"
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
#include "G4StepLimiter.hh"
#include "G4EmStandardPhysicsWVI.hh"
#include "G4eCoulombScatteringModel.hh"
#include "G4EmStandardPhysicsSS.hh"
#include "G4RunManager.hh"

MyPhysicsMessenger::MyPhysicsMessenger() {
    thetaLimitCmd = new G4UIcmdWithADoubleAndUnit("/msc/thetaLimit", this);
    thetaLimitCmd->SetGuidance("Set the MscThetaLimit parameter (in radians)");
    thetaLimitCmd->SetParameterName("thetaLimit", false);
    thetaLimitCmd->SetDefaultUnit("rad");
    thetaLimitCmd->SetRange("thetaLimit>=0.0");
}

MyPhysicsMessenger::~MyPhysicsMessenger() {
    delete thetaLimitCmd;
}

void MyPhysicsMessenger::SetNewValue(G4UIcommand* command, G4String newValue) {
    if (command == thetaLimitCmd) {
        G4double val = thetaLimitCmd->GetNewDoubleValue(newValue);
        G4EmParameters::Instance()->SetMscThetaLimit(val); // Set the theta limit
        G4cout << "Set MscThetaLimit to " << val << " radians" << G4endl;
    }
}

MyPhysicsList::MyPhysicsList() {
    new MyPhysicsMessenger();

    // Hardcode the desired physics list
    RegisterPhysics(new G4EmStandardPhysicsSS());
    
    // Configure step-cut parameterstask
    G4EmParameters* emParams = G4EmParameters::Instance();
    emParams->SetStepFunction(0.05, 1 * CLHEP::um);

    // Set global energy range for production cuts
    G4ProductionCutsTable::GetProductionCutsTable()->SetEnergyRange(0.000001 * CLHEP::eV, 100 * CLHEP::GeV);
}

MyPhysicsList::~MyPhysicsList() {
}

void MyPhysicsList::SetCuts() {
    G4VUserPhysicsList::SetCuts();

    // Set specific cuts for individual particles
    SetCutValue(0.1 * CLHEP::um, "gamma");  
    SetCutValue(0.1 * CLHEP::um, "e-");
    SetCutValue(0.1 * CLHEP::um, "e+");
    SetCutValue(1.0 * CLHEP::nm, "proton"); // very small range cut for protons (careful: increases CPU)

}

void MyPhysicsList::ConstructProcess() {
    // With the approach commented out other models could be used for further validation, like Urban and Wentzel_IV
    G4VModularPhysicsList::ConstructProcess();

    // --- Set Multiple Scattering Model for Protons ---
    /* 
    G4ParticleDefinition* proton = G4Proton::ProtonDefinition();
        G4ProcessManager* pmanager = proton->GetProcessManager();

    // Remove existing msc processes for protons
            G4int nProc = pmanager->GetProcessListLength();
            for (G4int i = nProc - 1; i >= 0; --i) {
                G4VProcess* proc = (*pmanager->GetProcessList())[i];
        if (proc->GetProcessName() == "msc") {
                    pmanager->RemoveProcess(proc);
                }
            }

    // // --- Urban Model (uncomment to use instead) ---
    auto urbanModel = new G4UrbanMscModel();
    auto mscUrban = new G4hMultipleScattering();
    mscUrban->AddEmModel(0, urbanModel);
    pmanager->AddProcess(mscUrban, -1, 1, 1);

    // auto stepLimiter = new G4StepLimiter();
    // pmanager->AddDiscreteProcess(stepLimiter);
    */

}
