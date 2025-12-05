#include "action.hh"
#include "generator.hh"
#include "runaction.hh"
#include "myeventaction.hh"
#include "mysteppingaction.hh"
#include "mytrackingaction.hh" 

MyActionInitialization::MyActionInitialization() {}
MyActionInitialization::~MyActionInitialization() {}

void MyActionInitialization::BuildForMaster() const {
    SetUserAction(new MyRunAction());
}

void MyActionInitialization::Build() const {
    SetUserAction(new MyPrimaryGenerator());
    SetUserAction(new MyRunAction());
    MySteppingAction* steppingAction = new MySteppingAction();
    SetUserAction(steppingAction);
    MyEventAction* eventAction = new MyEventAction(steppingAction);
    SetUserAction(eventAction);
    SetUserAction(new MyTrackingAction());
}

