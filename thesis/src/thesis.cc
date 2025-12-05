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
    runManager->SetNumberOfThreads(16); // Set the number of threads
#else
    G4RunManager* runManager = new G4RunManager();
#endif

    // --- allow selecting physics before MyPhysicsList is constructed ---
    std::string phys_token;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-phys" && i + 1 < argc) { phys_token = argv[i+1]; break; }
    }
    if (phys_token.empty()) {
        const char* e = std::getenv("G4_PHYSICS");
        if (e) phys_token = std::string(e);
    }
    if (!phys_token.empty()) {
        SetSelectedPhysics(phys_token);
    }

    // runManager->Initialize();

    G4UIExecutive* ui = 0;

    if(argc == 1)
    {
        ui = new G4UIExecutive(argc, argv);
    }
    
    G4VisManager* visManager = new G4VisExecutive();
    visManager->Initialize();

    G4UImanager* UImanager = G4UImanager::GetUIpointer();

    runManager->SetUserInitialization(new MyDetectorConstruction());
    runManager->SetUserInitialization(new MyPhysicsList());
    runManager->SetUserInitialization(new MyActionInitialization());

    if(ui)
    {
        UImanager->ApplyCommand("/control/execute vis.mac");
        ui->SessionStart();    
    }
    else
    {
        // Robust macro selection:
        // - skip recognized flags (e.g. -phys and its value)
        // - accept the first non-flag positional argument as the macro
        std::string macroFile;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "-phys") { ++i; continue; } // skip the -phys token and its value
            if (a.size() >= 2 && a.front() == '<' && a.back() == '>') {
                // user accidentally passed <macro>, strip angle brackets
                a = a.substr(1, a.size() - 2);
            }
            if (!a.empty() && a[0] != '-') { macroFile = a; break; }
        }
        if (macroFile.empty()) {
            G4cerr << "No macro file provided on command line." << G4endl;
        } else {
            G4String command = "/control/execute ";
            UImanager->ApplyCommand(command + macroFile);
        }
    } 


    delete visManager;
    delete runManager;

    return 0;
}

