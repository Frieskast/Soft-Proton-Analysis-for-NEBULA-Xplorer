#ifndef MYEVENTACTION_HH
#define MYEVENTACTION_HH

#include "G4UserEventAction.hh"
#include "G4Event.hh"
#include "mysteppingaction.hh" // <-- Include header

class MyEventAction : public G4UserEventAction {
public:
    // --- MODIFICATION: Add constructor and pointer member ---
    MyEventAction(MySteppingAction* steppingAction);
    virtual ~MyEventAction();
    
    virtual void BeginOfEventAction(const G4Event* event) override;
    virtual void EndOfEventAction(const G4Event* event) override;

private:
    MySteppingAction* fSteppingAction;
};

#endif