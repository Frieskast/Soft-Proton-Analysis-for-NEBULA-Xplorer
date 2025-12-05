#include "action.hh"
#include "mysteppingaction.hh"
#include "runaction.hh"

MyActionInitialization::MyActionInitialization()
{}

MyActionInitialization::~MyActionInitialization()
{}

void MyActionInitialization::BuildForMaster() const
{
	MyRunAction *runAction = new MyRunAction();
	SetUserAction(runAction);
}

void MyActionInitialization::Build() const
{	
	// Register the generator of protons
	MyPrimaryGenerator* generator = new MyPrimaryGenerator();
	SetUserAction(generator);

	// Register the stepping action
	MySteppingAction* steppingAction = new MySteppingAction();
	SetUserAction(steppingAction);

	SetUserAction(new MyRunAction());
}

