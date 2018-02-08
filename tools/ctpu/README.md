# CTPU: The Cloud TPU Provisioning Utility #

`ctpu` is an **experimental** tool that helps manage setting up a Cloud TPU. It
is focused on supporting data scientists using Cloud TPUs for their research and
model development. Check out the below for a walk through of how to use `ctpu`.

> Pro tip: power users and/or users wishing to perform parallel hyperparameter
> search should consider scripting their work or using tools such as GKE.

## Install `ctpu` ##

### Cloud Shell ###

The easiest way to start using Cloud TPUs with `ctpu` is via Google Cloud shell.
Click on the button below to follow a tutorial that will walk you through
getting everything set up.

[![Open in Cloud Shell](http://gstatic.com/cloudssh/images/open-btn.svg)](https://console.cloud.google.com/cloudshell/open?git_repo=https%3A%2F%2Fgithub.com%2Ftensorflow%2Ftpu&page=shell&tutorial=tools%2Fctpu%2Ftutorial.md)

### Local Machine ###

You can also use `ctpu` from your local machine. Follow the instructions below
to install and configure `ctpu` locally.

#### Download ####

Download `ctpu` with one of following commands:

 * **Linux**: `wget https://dl.google.com/cloud_tpu/ctpu/latest/linux/ctpu && chmod a+x ctpu`
 * **Mac**: `wget https://dl.google.com/cloud_tpu/ctpu/latest/darwin/ctpu && chmod a+x ctpu`
 * **Windows**: _Coming soon!_

#### Install ####

While you can use `ctpu` in your local directory (by prefixing all commands with
`./`; ex: `./ctpu config`), we recommend installing it somewhere on your
`$PATH`. (e.g. `cp ctpu ~/bin` to install for just yourself,
`sudo cp ctpu /usr/bin` for all users of your machine.)

#### Configuration ####

In order to use `ctpu` you need to provide it with a bit of additional
information.

 1. **Configure `gcloud` credentials**: If you have never used `gcloud` before,
    you will need to configure it. Run `gcloud auth login` to allocated
    credentials for `gcloud` to use when operating on your behalf.
 2. **Configure `ctpu` credentials**: `ctpu` uses the "application default"
    credentials set up by the Google SDK. In order to allocate your application
    default credentials, run: `gcloud auth application-default login`.
 3. **Configure project & zone** (optional): In order to avoid having to specify
    the GCP project and zone every time you run `ctpu`, you can set default
    values by running `gcloud config set project $PROJECT` and
    `gcloud config set compute/zone $ZONE`, substituting `$PROJECT` and `$ZONE`
    with your desired project and zone. If you are using multiple projects, you
    can group your default values using `gcloud config configurations`.

## Usage Instructions ##

There are 4 main subcommands to know when using `ctpu`:

 - **up**: `ctpu up` will create a GCE VM with TensorFlow pre-installed, and
   create a corresponding Cloud TPU. If necessary, it will enable the
   appropriate GCP APIs, and include
 - **pause**: `ctpu pause` will stop your GCE VM, and delete your Cloud TPU. Use
   this command when you'd like to go to lunch or when you're done for the night
   to save money. (No need to pay for a Cloud TPU or GCE VM if you're not using
   them.) When you're ready to get back going again, just run `ctpu up`, and
   you can pick back up right where you left off! *Note: you will still be
   charged for the disk space consumed by your GCE VM while it's paused.*
 - **delete**: `ctpu delete` will delete your GCE VM and Cloud TPU. Use this
   command if you're done using Cloud TPUs for a while or want to clean up your
   allocated resources.
 - **status**: `ctpu status` will query the GCP APIs to determine the current
   status of your Cloud TPU and GCE VM.

### Global Flags ###

There are a few flags common to all subcommands. These "global" flags are placed
before the subcommand. For example: `ctpu -name=saeta-2 config` (where
`-name=saeta2` is the global flag, and `config` is the subcommand). The most
common global flag is the `-name` flag. Use the `-name` flag when you'd like to
have multiple independent workspaces in the same GCP project, or if `ctpu`
doesn't guess a useful name. (`ctpu` defaults to naming your VM + TPU pair
(also referred to as a Cloud TPU flock) after your username.)

> Note: All flags can also be "double-dash" prefixed. (e.g. `--name=foo`)

#### Common GCE Configuration ####

While it's possible to use global flags to define the GCP project and GCE zone
you'd like to allocate your Cloud TPU and VMs in, it's often easier to use
`gcloud`'s built-in configuration system. If you didn't set a default
configuration when you installed gcloud, you can set (or reset) one using the
following commands:

```
gcloud config set project $MY_PROJECT
gcloud config set compute/zone us-central1-c
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

If you're having problems getting your credentials right, use the `ctpu config`
command to print out the configuration `ctpu` would use when creating your Cloud
TPU and GCE VM.

## Code Lab ##

This code lab walks through how to use the `ctpu` tool to use a Cloud TPU from
your local machine. (If you're using Cloud Shell, please follow the
[tutorial](https://console.cloud.google.com/cloudshell/open?git_repo=https%3A%2F%2Fgithub.com%2Ftensorflow%2Ftpu&page=shell&tutorial=tools%2Fctpu%2Ftutorial.md))

### Prerequisites ###

 - `ctpu` installed and configured. (See above.)
 - A GCP project with allocated TPU Quota.

### Check Configuration ###

Run `ctpu config` to view the name of the GCE VM and Cloud TPU `ctpu` will
create. For example:

```
saeta@saeta:~$ ctpu config
ctpu configuration:
        name: saeta
        project: ctpu-test-project
        zone: us-central1-c
```

### Bring up your flock ###

The most common way to use a Cloud TPU is from a GCE VM. You run your TensorFlow
python script on a GCE VM, and connect to a Cloud TPU over the network. `ctpu`
refers to the VM + TPU pair as a "flock". To allocate your flock, simply run:
`ctpu up`. `ctpu` will perform a number of actions on your behalf to get your
GCP environment properly setup and configured.

> Note: The first time you allocate a Cloud TPU, some additional operations may
> be required. Please do be patient. :-)

Once `ctpu up` has successfully allocated a Cloud TPU and a GCE VM with
TensorFlow installed, `ctpu` will automatically log you in to your GCE VM.

> Pro tip: If you need to reconnect to your GCE VM, just re-run `ctpu up`! If
> you'd like an additional ssh connection to your GCE VM and would like to get
> rid of the warnings (`bind: Address already in use` and
> `channel_setup_fwd_listener_tcpip: ...`), run `ctpu up --forward-ports=false`.

### Run a computation on a TPU ###

The GCE VM's that `ctpu` creates for you automatically have the latest stable
[TensorFlow](https://www.tensorflow.org/) version pre-installed. They also have
a few sample programs available at `/usr/share/tpu-demos/`

> TODO(saeta): Include running `mnist.py` on the Cloud TPU on the CLI

#### Using TensorBoard ####

When `ctpu` opens the ssh connection to your GCE VM, it also configures port
forwarding for the default TensorBoard port (6006). To use Tensorboard, run the
`tensorboard` command on the GCE VM, and then navigate to
[`localhost:6006`](http://localhost:6006/) to view TensorBoard.

> Pro tip: Use a terminal multiplexer such as `tmux` or `screen` to run your
> TensorFlow script and the `tensorboard` command at the same time.

<!-- TODO(saeta): Add an example invocation with model dir & logdir -->

#### Using Jupyter Notebooks ####

`ctpu` makes it easy to use [juypter notebooks](http://jupyter.org/) with Cloud
TPUs, because `ctpu` also opens up port forwarding for the default jupyter
notebook port (8888). Run `jupyter notebook` on the command line, and navigate
to [`localhost:8888`](http://localhost:8888/) on your local computer.

#### Connecting directly to the Cloud TPU ####

Power users may want to run code on their local machine that runs computations
on a Cloud TPU. `ctpu` also port-forwards `8470` and `8466` from the Cloud TPU
to your local machine. Therefore, you can run a TensorFlow script on your local
machine, and connect to `grpc://localhost:8470` and talk to your Cloud TPU.

> Note: running in this configuration incurs some additional performance
> overhead. Do not use this configuration when running benchmarks or other
> high performance workloads.

### Take a break ###

If you're done for the day, have to run to a meeting, or just would like to get
some coffee while you design your new machine learning algorithm, you can
"pause" your flock to save money. Just run `ctpu pause`.

When you pause your flock, you will de-allocate your Cloud TPU, and shut down
your GCE VM. All files saved on GCS or the local disk of your GCE VM will be
preserved.

To resume work, just run `ctpu up`!

### Cleaning up ###

To clean up your GCE VM and Cloud TPU, just run `ctpu delete`. This will ensure
you are not charged for your GCE VM or your Cloud TPU any further. _Important
note: you will still be charged for files stored in GCS._

### Multiple Flocks ###

By default, `ctpu` creates the flock name based on your username. We find this
to be the most common way for people to work. That said, if you're working on
multiple independent projects, you can configure the flock name on the command
line. In doing so, you can use multiple Cloud TPU flocks at the same time.

> Pro tip: If you're doing multiple independent experiments, you can often use
> a single GCE VM to drive training jobs on multiple Cloud TPUs concurrently.

## Security Documentation ##

The `ctpu` tool focuses on user egonomics, and thus automatically selects
reasonable defaults that are expected to work for the majority of users. We
document these choices that are potentially security related here as well as
how to customize the security posture.

 - **Port Forwarding**: In order to make tools like [`tensorboard`](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard)
   work out of the box, `ctpu` automatically configures port forwarding over the
   ssh tunnel to your GCE VM. If you'd like to disable port forarding, add the
   `--forward-ports=false` flag to `ctpu up`. Example:

   ```
   ctpu up --forward-ports=false
   ```

 - **IAM & Service Management**: A Cloud TPU typically reads data from (and
   saves checkpoints to) [GCS](https://cloud.google.com/storage/docs/). A Cloud
   TPU also outputs logs to
   [Stackdriver Logging](https://cloud.google.com/logging/). By default, Cloud
   TPUs have no permissions on your project. The `ctpu` tool automatically sets
   up the Cloud TPU's permissions to output TensorFlow logs to your project, and
   allows the Cloud TPU to read all storage buckets in our project. However, if
   `ctpu` sees that your Cloud TPU already has _some_ access pre-configured, it
   will make no changes.

 - **SSH Agent Forwarding**: When ssh-ing into the GCE VM, `ctpu` supports SSH
   Agent forwarding. When working with non-public repositories (e.g. private
   GitHub repositories), credentials are required to clone the source tree. SSH
   Agent forwarding allows users to forward their credentials from their local
   machine to the GCE VM to avoid persisting credentials on the GCE VM. If you
   would like to disable SSH Agent Forwarding, pass the `--forward-agent=false`
   flag when executing `ctpu up`. Example:

   ```
   ctpu up --forward-agent=false
   ```

## Current limitations of ctpu ##

 - **Multiple Accounts**: `ctpu` cannot correctly handle if you use multiple
   Google accounts across different projects. (e.g. `alice@example.com` for work
   and `alice@gmail.com` for personal development.)
 - **Name restrictions**: In order to prevent clashes, we require that all
   flock names are longer than 2 characters. If your username is 2 characters or
   less, you will have to manually set a flock name on the command line with the
   `-name` global flag.

### Stability ###

`ctpu` is an *experimental* tool and thus is not guaranteed to be stable,
including, but not limited to, the following ways:

 - **TF version**: When `ctpu` creates a Cloud TPU and GCE VM, it will create it
   with the latest stable TensorFlow version. As new TensorFlow versions are
   released, you will need to upgrade the installed TensorFlow on your VMs, or
   delete your GCE VM (after appropriately saving your work!) and re-create it
   using `ctpu up`.
 - **Commands & Output**: Do not rely on the presence of particular subcommands,
   flags, or the format of their output. It can change without warning.
 - **The existence of ctpu itself**: It is entirely possible that `ctpu` as it
   exists today will be gone tomorrow!

## Contributing ##

_Contributions are welcome to the `ctpu` tool!_

### Bug Reports ###

If you encounter a reproducible issue with `ctpu`, please do file a bug report!
It will be most helpful if you include:

 1. The full output when running the command with the `-log-http` global flag
    set to true
 2. The output of `ctpu config`, `ctpu version`, and `ctpu list` both before
    and after the failing command.
 3. Steps to reproduce the issue on a clean GCP project.

### Developing ###

The code is layed out in the following packages:

 - **`config`**: This package contains the tool-wide configuration, such as (1)
   the credentials used to communicate with GCP, (2) desired zone, and (3) the
   desired flock name.
 - **`ctrl`**: This package contains the thin wrappers around the
   [Google API Go SDK](https://github.com/google/google-api-go-client). For
   details on the SDK, see the godocs for
   [GCE](https://godoc.org/google.golang.org/api/compute/v1) and
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

Once you're in this directory, you can use `go build` and `go test`.

For additional background on standard `go` idioms, check out:

 - [How to Write Go Code](https://golang.org/doc/code.html)
 - [Effective Go](https://golang.org/doc/effective_go.html)
 - [Go FAQ](https://golang.org/doc/faq)
