# Commons - Shared libraries and utilities

This directory contains shared libraries for the trainers and models. Because
there is such a diversity in the tools required for models, we generally try to
put that code in the models themselves.

**Duplication is okay most of the time**, but in the rare cases where there is a
piece of code  that needs to be shared by many trainers or models, this is a
good place to put that logic.

Only libraries with consistent interfaces that function across a wide array of
`models` and `trainers` should be introduced into this directory so that we don't
create unnecessary dependencies between models and trainers.
