#include <iostream>
#include <fstream>
#include "G4MTRunManager.hh"
#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4VisManager.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include "physics.hh"
#include "detector.hh"
#include "action.hh"
#include "G4VExceptionHandler.hh"
#include "G4ios.hh"
#include "construction.hh"
#include "G4UserLimits.hh"
#include "runaction.hh"


// This is the main hub of the simulation

class MyExceptionHandler : public G4VExceptionHandler {
public:
    G4bool Notify(const char* originOfException, const char* exceptionCode,
        G4ExceptionSeverity severity, const char* description) override {
        G4cerr << "\n*** Custom G4Exception Handler ***" << G4endl;
        G4cerr << "Exception from: " << originOfException << G4endl;
        G4cerr << "Code: " << exceptionCode << G4endl;
        G4cerr << "Severity: " << severity << G4endl;
        G4cerr << "Description: " << description << G4endl;

        if (severity == FatalException) {
            G4cerr << "\nFatal error encountered! Pausing instead of exiting...\n";
            G4cerr << "Press Enter to continue." << G4endl;
            std::cin.get();  // Wait for user input before continuing
            return true;  // Continue execution instead of aborting
        }
        return false;  // Use default behavior for non-fatal exceptions
    }
};

int main(int argc, char** argv) {
#ifdef G4MULTITHREADED
    G4MTRunManager* runManager = new G4MTRunManager();
    runManager->SetNumberOfThreads(15); // Set the number of threads
#else
    G4RunManager* runManager = new G4RunManager();
#endif

    runManager->SetUserInitialization(new MyDetectorConstruction());

    runManager->SetUserInitialization(new MyPhysicsList());

    runManager->SetUserInitialization(new MyActionInitialization());

    G4UIExecutive* ui = nullptr;
    if (argc == 1) {
        ui = new G4UIExecutive(argc, argv);
    }

    G4VisManager* visManager = new G4VisExecutive();
    visManager->Initialize();

    G4UImanager* UImanager = G4UImanager::GetUIpointer();

    if (ui) {
        UImanager->ApplyCommand("/control/execute vis.mac");
        ui->SessionStart();
    } else {
        G4String command = "/control/execute ";
        G4String fileName = argv[1];
        UImanager->ApplyCommand(command + fileName);
    }

    delete visManager;
    delete runManager;

    return 0;
}
