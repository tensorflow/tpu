# CTPU: The Cloud TPU Provisioning Utility #

`ctpu` is a tool that helps you set up a Cloud TPU. It
is focused on supporting data scientists using Cloud TPUs for their research and
model development.

There are 4 main subcommands to know when using `ctpu`:

 - **status**: `ctpu status` will query the GCP APIs to determine the current
   status of your Cloud TPU and Compute Engine VM.
 - **up**: `ctpu up` will create a Compute Engine VM with TensorFlow
   pre-installed, and create a corresponding Cloud TPU. If necessary, it will
   enable the appropriate GCP APIs, and configure default access levels.
   Finally, it will `ssh` into your Compute Engine VM so you're all ready to
   start developing! The environment variable `$TPU_NAME` is set automatically.
 - **pause**: `ctpu pause` will stop your Compute Engine VM, and delete your
   Cloud TPU. Use this command when you'd like to go to lunch or when you're
   done for the night to save money. (No need to pay for a Cloud TPU or
   Compute Engine VM if you're not using them.) When you're ready to get back
   going again, just run `ctpu up`, and you can pick back up right where you
   left off! *Note: you will still be charged for the disk space consumed by
   your Compute Engine VM while it's paused.*
 - **delete**: `ctpu delete` will delete your Compute Engine VM and Cloud TPU.
   Use this command if you're done using Cloud TPUs for a while or want to
   clean up your allocated resources.

> Pro tip: `ctpu` makes simplifying assumptions on your behalf and thus may not
> be suitable for power users. For example, if you're executing a parallel
> hyperparameter search, consider scripting calls to `gcloud` instead.

## Install `ctpu` ##

You can get started using `ctpu` in one of two ways:

1. Using Google Cloud Shell (**recommended**). This is the fastest and easiest
   way to get started, and comes with a tutorial to walk you through all the
   steps.
2. Using your local machine. You can download and run `ctpu` on your local
   machine

Follow the appropriate instructions below to get started.

### Cloud Shell ###

Click on the button below to follow a tutorial that will walk you through
getting everything set up.

[![Open in Cloud Shell](http://gstatic.com/cloudssh/images/open-btn.svg)](https://console.cloud.google.com/cloudshell/open?git_repo=https%3A%2F%2Fgithub.com%2Ftensorflow%2Ftpu&page=shell&tutorial=tools%2Fctpu%2Ftutorial.md)

Note: The above request clones the `ctpu` repository into your Cloud Shell. The
only reason for cloning the repo is so that you can view the tutorial in the
shell. The `ctpu` tool itself is pre-installed on the Cloud Shell.

### Local Machine ###

Alternatively, you can also use `ctpu` from your local machine. Follow the
instructions below to install and configure `ctpu` locally.

#### Download ####

Download `ctpu` with one of following commands:

 * **Linux**: `wget https://dl.google.com/cloud_tpu/ctpu/latest/linux/ctpu && chmod a+x ctpu`
 * **Mac**: `curl -O https://dl.google.com/cloud_tpu/ctpu/latest/darwin/ctpu && chmod a+x ctpu`
 * **Windows**: _Coming soon!_

#### Install ####

While you can use `ctpu` in your local directory (by prefixing all commands with
`./`; example: `./ctpu print-config`), we recommend installing it somewhere on
your `$PATH`. (example: `cp ctpu ~/bin/` to install for just yourself, or
`sudo cp ctpu /usr/bin/` for all users of your machine.)

#### Configuration ####

In order to use `ctpu` you need to provide it with a bit of additional
information.

 1. **Configure `gcloud` credentials**: If you have never used `gcloud` before,
    you will need to configure it. Run `gcloud auth login` to allocate
    credentials for `gcloud` to use when operating on your behalf.
 2. **Configure `ctpu` credentials**: `ctpu` uses the "application default"
    credentials set up by the Google SDK. In order to allocate your application
    default credentials, run: `gcloud auth application-default login`.

## Usage Details ##

### Common Global Flags ###

There are a few flags common to all subcommands. These "global" flags can
be placed before or after the subcommand.
For example: `ctpu -name=saeta-2 print-config` or
`ctpu print-config -name=saeta-2`(where
`-name=saeta2` is the global flag and `print-config` is the subcommand). The most
commonly used global flags are the `-name` flag and the `-zone` flag.

> Note: All flags can also be "double-dash" prefixed. (e.g. `--name=foo`)

* `-name` - Specifies the name of your Cloud TPU. Use the `-name` flag when
  you'd like to have multiple independent workspaces in the same GCP project, or
  if `ctpu` doesn't automatically assign a useful name. Note: `ctpu` defaults to
  naming your VM + TPU pair after your username. (The VM + TPU pair is also
  called a Cloud TPU flock.)
* `-zone` - Specifies the Compute Engine zone. The default zone for `ctpu` is
  `us-central1-b`.

> Note: The effect of a global flag is scoped to the current invocation of the
  `ctpu` command. You must specify the global flag each time you run the command.
  For example, assume you want to create your Cloud TPU in zone `us-central1-c`
  and you therefore run `ctpu up -zone=us-central1-c`. The next time you
  run a `ctpu` command, you must specify the zone again, otherwise `ctpu` will
  reset its configuration to the default zone. So, for example, if you run
  `ctpu status`, the configuration zone for `ctpu` will revert to the default
  `us-central1-b`. If you want to create another Cloud TPU in `us-central1-c`,
  you must run `ctpu up -zone=us-central1-c` again. If you're enrolled in the
  [TFRC program](https://www.tensorflow.org/tfrc/) you must run your TPUs
  in zone **us-central1-f**.

As an alternative to global flags for project and zone, consider the built-in
configuration system for `gcloud`, described below.

#### Using the gcloud Configuration System ####

While it's possible to use global flags on the `ctpu` command to define
the GCP project and Compute Engine zone you'd like to allocate your
Cloud TPU and VMs in, it's often easier to use `gcloud`'s built-in
configuration system. If you didn't set a
default configuration when you installed gcloud, you can set (or reset) one
using the following commands:

```
gcloud config set project $MY_PROJECT
gcloud config set compute/zone us-central1-b
gcloud config set compute/region us-central1
```

If you'd like to maintain multiple independent configurations (e.g you're
using GCP for a personal project, and a project at work), you can use the
`gcloud config configurations` subcommand to manage multiple independent
configurations. `ctpu` will use the currently active configuration
automatically.

### Getting help ###

If you're ever confused on how to use the `ctpu` tool, you can always run
`ctpu help` to get a print out of the major usage documentation. If you'd like
to learn more about a particular subcommand, run `ctpu help $SUBCOMMAND` (for
example: `ctpu help up`). If you'd simply like a list of all the available
subcommands, simply execute `ctpu commands`.

If you're having problems getting your credentials right, use the
`ctpu print-config` command to print out the configuration `ctpu` would use when
creating your Cloud TPU and Compute Engine VM.

## Security Documentation ##

The `ctpu` tool focuses on user egonomics, and thus automatically selects
reasonable defaults that are expected to work for the majority of users. We
document these choices that are potentially security related here as well as
how to customize the security posture.

 - **Port Forwarding**: In order to make tools like [`tensorboard`](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard)
   work out of the box, `ctpu` automatically configures port forwarding over the
   ssh tunnel to your Compute Engine VM. If you'd like to disable port
   forarding, add the `--forward-ports=false` flag to `ctpu up`. Example:

   ```
   ctpu up --forward-ports=false
   ```

 - **IAM & Service Management**: A Cloud TPU typically reads data from (and
   saves checkpoints to)
   [Cloud Storage](https://cloud.google.com/storage/docs/). A Cloud
   TPU also outputs logs to
   [Stackdriver Logging](https://cloud.google.com/logging/). By default, Cloud
   TPUs have no permissions on your project. The `ctpu` tool automatically sets
   up the Cloud TPU's permissions to output TensorFlow logs to your project, and
   allows the Cloud TPU to read all storage buckets in our project. However, if
   `ctpu` sees that your Cloud TPU already has _some_ access pre-configured, it
   will make no changes.

 - **SSH Agent Forwarding**: When ssh-ing into the Compute Engine VM, `ctpu`
   supports SSH Agent forwarding. When working with non-public repositories
   (e.g. private GitHub repositories), credentials are required to clone the
   source tree. SSH Agent forwarding allows users to forward their credentials
   from their local machine to the Compute Engine VM to avoid persisting
   credentials on the Compute Engine VM. If you would like to disable SSH Agent
   Forwarding, pass the `--forward-agent=false` flag when executing `ctpu up`.
   Example:

   ```
   ctpu up --forward-agent=false
   ```

## Current limitations of ctpu ##

 - **Multiple Accounts**: `ctpu` cannot correctly handle if you use multiple
   Google accounts across different projects. (e.g. `alice@example.com` for work
   and `alice@gmail.com` for personal development.) Instead, please use `ctpu`
   in Google Cloud Shell where you will have a different shell environment for
   each account.
 - **Name restrictions**: In order to prevent clashes, we require that all
   flock names are longer than 2 characters. If your username is 2 characters or
   less, you will have to manually set a flock name on the command line with the
   `-name` global flag.
 - **TF version**: When `ctpu` creates a Cloud TPU and Compute Engine VM, it
   creates the VM with the latest stable TensorFlow version. When new TensorFlow
   versions are released, you must upgrade the installed TensorFlow on your VMs,
   or delete your Compute Engine VM (after appropriately saving your work!) and
   re-create it using `ctpu up`.

## Contributing ##

_Contributions are welcome to the `ctpu` tool!_

### Bug Reports ###

If you encounter a reproducible issue with `ctpu`, please do file a bug report!
It will be most helpful if you include:

 1. The full output when running the command with the `-log-http` global flag
    set to true
 2. The output of `ctpu print-config`, `ctpu version`, and `ctpu list` both before
    and after the failing command.
 3. Steps to reproduce the issue on a clean GCP project.

### Developing ###

The code is laid out in the following packages:

 - **`config`**: This package contains the tool-wide configuration, such as (1)
   the credentials used to communicate with GCP, (2) desired zone, and (3) the
   desired flock name.
 - **`ctrl`**: This package contains the thin wrappers around the
   [Google API Go SDK](https://github.com/google/google-api-go-client). For
   details on the SDK, see the godocs for
   [Compute Engine](https://godoc.org/google.golang.org/api/compute/v1) and
   [Cloud TPUs](https://godoc.org/google.golang.org/api/tpu/v1alpha1).
 - **`commands`**: This package contains the business logic for all subcommands.
 - **`main`**: The main package ties everything together.

In order to keep the code organized, dependencies are only allowed on packages
above the current package in the list. Concretely, the `commands` package can
depend on `ctrl` and `config`, but `config` cannot depend on `ctrl`.

Contributed code must conform to the Golang style guide, and follow Go best
practices. Additionally, all contributions should include unit tests in order
to ensure there are no regressions in functionality in the future. Unit tests
must not depend on anything in the environment, and must not make any network
connections.

#### Developer Workflow ####

`ctpu` is developed as a standard [go](https://golang.org/) project. To check
out the code for development purposes, execute:

```
go get -t github.com/tensorflow/tpu/tools/ctpu/...
go test github.com/tensorflow/tpu/tools/ctpu/...
```

When you're in this directory, you can use `go build` and `go test`.

For additional background on standard `go` idioms, check out:

 - [How to Write Go Code](https://golang.org/doc/code.html)
 - [Effective Go](https://golang.org/doc/effective_go.html)
 - [Go FAQ](https://golang.org/doc/faq)
