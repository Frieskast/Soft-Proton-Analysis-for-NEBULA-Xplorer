#ifndef MYTRACKINGACTION_HH
#define MYTRACKINGACTION_HH

#include "G4UserTrackingAction.hh"

class MyTrackingAction : public G4UserTrackingAction {
public:
    MyTrackingAction();
    virtual ~MyTrackingAction();
    virtual void PreUserTrackingAction(const G4Track* track) override;
    virtual void PostUserTrackingAction(const G4Track* track) override;
};

#endif