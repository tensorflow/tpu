/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include "gasket_interrupt.h"
#include <linux/interrupt.h>
#include <linux/rwlock.h>
#include "gasket_constants.h"
#include "gasket_logging.h"
#include "gasket_sysfs.h"
#define CREATE_TRACE_POINTS 
#include <trace/events/gasket_interrupt.h>
#define MSIX_RETRY_COUNT 3
struct gasket_interrupt_data {
 char **names;
 struct pci_dev *pci_dev;
 int msix_configured;
 int num_msix_interrupts;
 int num_interrupts;
 const struct legacy_gasket_interrupt_desc *legacy_interrupts;
 const struct gasket_interrupt_desc *interrupts;
 int legacy_interrupt_bar_index;
 int legacy_pack_width;
 int num_configured;
 bool *configured_interrupts;
 int *mapped_registers;
 bool *configured_registers;
 struct msix_entry *msix_entries;
 struct eventfd_ctx **eventfd_ctxs;
 rwlock_t eventfd_ctx_lock;
 ulong *interrupt_counts;
};
static bool gasket_interrupt_is_legacy(
 struct gasket_interrupt_data *interrupt_data)
{
 if (interrupt_data->legacy_interrupts != NULL)
  return true;
 return false;
}
static ssize_t interrupt_sysfs_show(
 struct device *device, struct device_attribute *attr, char *buf);
static irqreturn_t gasket_interrupt_handler(int irq, void *dev_id);
enum interrupt_sysfs_attribute_type {
 ATTR_INTERRUPT_COUNTS,
};
static struct gasket_sysfs_attribute interrupt_sysfs_attrs[] = {
 GASKET_SYSFS_RO(
  interrupt_counts, interrupt_sysfs_show, ATTR_INTERRUPT_COUNTS),
 GASKET_END_OF_ATTR_ARRAY,
};
static int gasket_interrupt_setup(struct gasket_dev *gasket_dev);
static int gasket_interrupt_msix_init(
 struct gasket_interrupt_data *interrupt_data);
static int legacy_gasket_interrupt_msix_init(
 struct gasket_dev *gasket_dev);
static void gasket_interrupt_msix_teardown(
 struct gasket_interrupt_data *interrupt_data);
static int gasket_interrupt_configure(
 struct gasket_dev *gasket_dev, int interrupt);
int gasket_interrupt_allocate(
 struct gasket_dev *gasket_dev, const char *name)
{
 int i, ret;
 struct gasket_interrupt_data *interrupt_data =
  gasket_dev->interrupt_data;
 int num_msix_interrupts = interrupt_data->num_msix_interrupts;
 interrupt_data->names = kcalloc(
  num_msix_interrupts, sizeof(char *), GFP_KERNEL);
 if (!interrupt_data->names)
  return -ENOMEM;
 for (i = 0; i < num_msix_interrupts; i++) {
  int name_bytes = strlen(name) + 8;
  interrupt_data->names[i] = kmalloc(name_bytes, GFP_KERNEL);
  if (!interrupt_data->names[i]) {
   ret = -ENOMEM;
   goto fail_name_allocate;
  }
  snprintf(interrupt_data->names[i], name_bytes, "%s%d-%d", name,
   gasket_dev->dev_idx, i);
 }
 interrupt_data->msix_entries = kcalloc(
  num_msix_interrupts, sizeof(struct msix_entry), GFP_KERNEL);
 if (!interrupt_data->msix_entries) {
  ret = -ENOMEM;
  goto fail_name_allocate;
 }
 interrupt_data->eventfd_ctxs = kcalloc(
  num_msix_interrupts, sizeof(struct eventfd_ctx *), GFP_KERNEL);
 if (!interrupt_data->eventfd_ctxs) {
  ret = -ENOMEM;
  goto fail_eventfd_allocate;
 }
 interrupt_data->interrupt_counts = kzalloc(
  sizeof(ulong) * num_msix_interrupts, GFP_KERNEL);
 if (!interrupt_data->interrupt_counts) {
  ret = -ENOMEM;
  goto fail_counts_allocate;
 }
 interrupt_data->configured_interrupts = kcalloc(
  num_msix_interrupts, sizeof(bool), GFP_KERNEL);
 if (!interrupt_data->configured_interrupts) {
  ret = -ENOMEM;
  goto fail_configured_interrupt_allocate;
 }
 rwlock_init(&interrupt_data->eventfd_ctx_lock);
 return 0;
fail_configured_interrupt_allocate:
 kfree(interrupt_data->interrupt_counts);
fail_counts_allocate:
 kfree(interrupt_data->eventfd_ctxs);
fail_eventfd_allocate:
 kfree(interrupt_data->msix_entries);
fail_name_allocate:
 for (i = 0; i < num_msix_interrupts; i++)
  kfree(interrupt_data->names[i]);
 kfree(interrupt_data->names);
 return ret;
}
int gasket_interrupt_init(struct gasket_dev *gasket_dev,
 const char *name,
 const struct gasket_interrupt_desc *interrupts,
 int num_interrupts,
 int num_msix_interrupts)
{
 int ret;
 struct gasket_interrupt_data *interrupt_data;
 interrupt_data = kzalloc(
  sizeof(struct gasket_interrupt_data), GFP_KERNEL);
 if (!interrupt_data)
  return -ENOMEM;
 gasket_dev->interrupt_data = interrupt_data;
 interrupt_data->pci_dev = gasket_dev->pci_dev;
 interrupt_data->num_msix_interrupts = num_msix_interrupts;
 interrupt_data->num_interrupts = num_interrupts;
 interrupt_data->interrupts = interrupts;
 interrupt_data->legacy_interrupts = NULL;
 interrupt_data->legacy_interrupt_bar_index = 0;
 interrupt_data->legacy_pack_width = 0;
 interrupt_data->num_configured = 0;
 ret = gasket_interrupt_allocate(gasket_dev, name);
 if (ret)
  goto fail_interrupt_allocate;
 interrupt_data->mapped_registers =
  kzalloc(sizeof(int) * num_msix_interrupts, GFP_KERNEL);
 if (!interrupt_data->mapped_registers) {
  ret = -ENOMEM;
  goto fail_mapped_registers_allocate;
 }
 interrupt_data->configured_registers =
  kzalloc(sizeof(bool) * num_interrupts, GFP_KERNEL);
 if (!interrupt_data->configured_registers) {
  ret = -ENOMEM;
  goto fail_configured_registers_allocate;
 }
 ret = gasket_interrupt_msix_init(interrupt_data);
 if (ret) {
  gasket_log_warn(gasket_dev, "Couldn't init msix: %d", ret);
  goto fail_msix_init;
 }
 ret = gasket_interrupt_setup(gasket_dev);
 if (ret) {
  gasket_log_warn(
   gasket_dev, "Couldn't setup interrupts: %d", ret);
  goto fail_msix_init;
 }
 gasket_sysfs_create_entries(
  &gasket_dev->accel_dev.dev, interrupt_sysfs_attrs);
 return 0;
fail_msix_init:
 kfree(interrupt_data->configured_registers);
fail_configured_registers_allocate:
 kfree(interrupt_data->mapped_registers);
fail_mapped_registers_allocate:
 gasket_interrupt_cleanup(gasket_dev);
fail_interrupt_allocate:
 kfree(interrupt_data);
 return ret;
}
int legacy_gasket_interrupt_init(struct gasket_dev *gasket_dev,
 const char *name, const struct legacy_gasket_interrupt_desc *interrupts,
 int num_interrupts, int pack_width, int bar_index)
{
 int ret;
 struct gasket_interrupt_data *interrupt_data;
 interrupt_data =
  kzalloc(sizeof(struct gasket_interrupt_data), GFP_KERNEL);
 if (!interrupt_data)
  return -ENOMEM;
 gasket_dev->interrupt_data = interrupt_data;
 interrupt_data->pci_dev = gasket_dev->pci_dev;
 interrupt_data->num_msix_interrupts = num_interrupts;
 interrupt_data->num_interrupts = num_interrupts;
 interrupt_data->interrupts = NULL;
 interrupt_data->legacy_interrupts = interrupts;
 interrupt_data->legacy_interrupt_bar_index = bar_index;
 interrupt_data->legacy_pack_width = pack_width;
 interrupt_data->num_configured = 0;
 ret = gasket_interrupt_allocate(gasket_dev, name);
 if (ret)
  goto fail_allocate;
 ret = legacy_gasket_interrupt_msix_init(gasket_dev);
 if (ret) {
  gasket_log_warn(gasket_dev, "Couldn't init msix: %d", ret);
  goto fail_msix_init;
 }
 ret = legacy_gasket_interrupt_setup(gasket_dev);
 if (ret) {
  gasket_log_warn(
   gasket_dev, "Couldn't setup interrupts: %d", ret);
  goto fail_msix_init;
 }
 gasket_sysfs_create_entries(
  &gasket_dev->accel_dev.dev, interrupt_sysfs_attrs);
 return 0;
fail_msix_init:
 gasket_interrupt_cleanup(gasket_dev);
fail_allocate:
 kfree(interrupt_data);
 return ret;
}
static int gasket_interrupt_msix_init(
 struct gasket_interrupt_data *interrupt_data)
{
 int ret = 1;
 int i;
 for (i = 0; i < interrupt_data->num_msix_interrupts; i++) {
  interrupt_data->msix_entries[i].entry = i;
  interrupt_data->msix_entries[i].vector = 0;
  interrupt_data->eventfd_ctxs[i] = NULL;
 }
 for (i = 0; i < MSIX_RETRY_COUNT && ret != 0; i++)
  ret = pci_enable_msix_exact(interrupt_data->pci_dev,
   interrupt_data->msix_entries,
   interrupt_data->num_msix_interrupts);
 if (ret)
  return ret > 0 ? -EBUSY : ret;
 interrupt_data->msix_configured = 1;
 return 0;
}
static int legacy_gasket_interrupt_msix_init(
 struct gasket_dev *gasket_dev)
{
 int ret = 0;
 int i;
 ret = gasket_interrupt_msix_init(gasket_dev->interrupt_data);
 if (ret)
  return ret;
 for (i = 0; i < gasket_dev->interrupt_data->num_interrupts; i++) {
  ret = gasket_interrupt_configure(gasket_dev, i);
  if (ret)
   goto fail_configure;
 }
 return 0;
fail_configure:
 gasket_interrupt_msix_teardown(gasket_dev->interrupt_data);
 return ret;
}
static int gasket_interrupt_clear_eventfd(
 struct gasket_dev *gasket_dev, struct gasket_filp_data *filp_data,
 int interrupt)
{
 struct gasket_interrupt_data *interrupt_data =
  gasket_dev->interrupt_data;
 ulong flags;
 if (filp_data && gasket_dev->driver_desc->interrupt_permissions_cb) {
  int retval = gasket_dev->driver_desc->interrupt_permissions_cb(
   filp_data, interrupt);
  if (retval < 0)
   return retval;
 }
 write_lock_irqsave(&interrupt_data->eventfd_ctx_lock, flags);
 if (interrupt_data->eventfd_ctxs[interrupt] != NULL)
  eventfd_ctx_put(interrupt_data->eventfd_ctxs[interrupt]);
 interrupt_data->eventfd_ctxs[interrupt] = NULL;
 write_unlock_irqrestore(&interrupt_data->eventfd_ctx_lock, flags);
 return 0;
}
static void gasket_interrupt_free_irq(
 struct gasket_interrupt_data *interrupt_data,
 int interrupt)
{
 if (!interrupt_data->configured_interrupts[interrupt])
  return;
 free_irq(interrupt_data->msix_entries[interrupt].vector,
  interrupt_data);
 interrupt_data->configured_interrupts[interrupt] = false;
 if (interrupt_data->num_configured == 0) {
  gasket_nodev_error(
   "num_configured already 0 when freeing interrupt %d",
    interrupt);
  return;
 }
 interrupt_data->num_configured--;
}
static void gasket_interrupt_clear_all_eventfds(struct gasket_dev *gasket_dev)
{
 struct gasket_interrupt_data *interrupt_data =
  gasket_dev->interrupt_data;
 int i;
 for (i = 0; i < interrupt_data->num_msix_interrupts; i++)
  gasket_interrupt_clear_eventfd(gasket_dev, NULL, i);
}
static void gasket_interrupt_msix_teardown(
 struct gasket_interrupt_data *interrupt_data)
{
 int i;
 for (i = 0; i < interrupt_data->num_msix_interrupts; i++)
  gasket_interrupt_free_irq(interrupt_data, i);
 interrupt_data->num_configured = 0;
 if (interrupt_data->msix_configured)
  pci_disable_msix(interrupt_data->pci_dev);
 interrupt_data->msix_configured = 0;
}
int gasket_interrupt_reinit(struct gasket_dev *gasket_dev)
{
 int ret;
 if (!gasket_dev->interrupt_data) {
  gasket_log_error(gasket_dev,
   "Attempted to reinit uninitialized interrupt data.");
  return -EINVAL;
 }
 gasket_interrupt_clear_all_eventfds(gasket_dev);
 gasket_interrupt_msix_teardown(gasket_dev->interrupt_data);
 if (gasket_interrupt_is_legacy(gasket_dev->interrupt_data))
  ret = legacy_gasket_interrupt_msix_init(
   gasket_dev);
 else
  ret = gasket_interrupt_msix_init(gasket_dev->interrupt_data);
 if (ret) {
  gasket_log_warn(gasket_dev, "Couldn't init msix: %d", ret);
  return 0;
 }
 if (gasket_interrupt_is_legacy(gasket_dev->interrupt_data))
  ret = legacy_gasket_interrupt_setup(gasket_dev);
 else
  ret = gasket_interrupt_setup(gasket_dev);
 return ret;
}
EXPORT_SYMBOL(gasket_interrupt_reinit);
int gasket_interrupt_reset_counts(struct gasket_dev *gasket_dev)
{
 gasket_log_debug(gasket_dev, "Clearing interrupt counts.");
 memset(gasket_dev->interrupt_data->interrupt_counts, 0,
  gasket_dev->interrupt_data->num_msix_interrupts *
   sizeof(*gasket_dev->interrupt_data->interrupt_counts));
 return 0;
}
static int gasket_interrupt_configure(
 struct gasket_dev *gasket_dev, int interrupt)
{
 int ret;
 struct gasket_interrupt_data *interrupt_data =
  gasket_dev->interrupt_data;
 if (interrupt_data->configured_interrupts[interrupt]) {
  gasket_log_error(gasket_dev,
   "Interrupt %d is already configured", interrupt);
  return -EINVAL;
 }
 ret = request_irq(interrupt_data->msix_entries[interrupt].vector,
  gasket_interrupt_handler, 0, interrupt_data->names[interrupt],
  interrupt_data);
 if (ret) {
  gasket_log_error(gasket_dev,
   "Cannot get IRQ for interrupt %d, vector %d; %d\n",
   interrupt,
   interrupt_data->msix_entries[interrupt].vector, ret);
  return ret;
 }
 interrupt_data->configured_interrupts[interrupt] = true;
 interrupt_data->num_configured++;
 return 0;
}
int legacy_gasket_interrupt_setup(struct gasket_dev *gasket_dev)
{
 int i;
 int pack_shift;
 ulong mask;
 ulong value;
 struct gasket_interrupt_data *interrupt_data =
  gasket_dev->interrupt_data;
 if (!interrupt_data) {
  gasket_log_error(
   gasket_dev, "Interrupt data is not initialized.");
  return -EINVAL;
 }
 if (!gasket_interrupt_is_legacy(interrupt_data)) {
  gasket_log_error(gasket_dev,
   "Unable to run legacy interrupt setup on device.");
  return -EPERM;
 }
 gasket_log_debug(gasket_dev, "Running legacy interrupt setup.");
 for (i = 0; i < interrupt_data->num_interrupts; i++) {
  gasket_log_debug(gasket_dev,
   "Setting up interrupt index %d with index 0x%llx and packing %d",
   interrupt_data->legacy_interrupts[i].index,
   interrupt_data->legacy_interrupts[i].reg,
   interrupt_data->legacy_interrupts[i].packing);
  if (interrupt_data->legacy_interrupts[i].packing == UNPACKED) {
   value = interrupt_data->legacy_interrupts[i].index;
  } else {
   switch (interrupt_data->legacy_interrupts[i].packing) {
   case PACK_0:
    pack_shift = 0;
    break;
   case PACK_1:
    pack_shift = interrupt_data->legacy_pack_width;
    break;
   case PACK_2:
    pack_shift =
     2 * interrupt_data->legacy_pack_width;
    break;
   case PACK_3:
    pack_shift =
     3 * interrupt_data->legacy_pack_width;
    break;
   default:
    gasket_nodev_error(
     "Found interrupt description with unknown enum %d",
     interrupt_data->legacy_interrupts[i]
      .packing);
    return -EINVAL;
   }
   mask = ~(0xFFFF << pack_shift);
   value = gasket_dev_read_64(gasket_dev,
     interrupt_data
      ->legacy_interrupt_bar_index,
     interrupt_data->legacy_interrupts[i]
      .reg) &
    mask;
   value |= interrupt_data->legacy_interrupts[i].index
     << pack_shift;
  }
  gasket_dev_write_64(gasket_dev, value,
   interrupt_data->legacy_interrupt_bar_index,
   interrupt_data->legacy_interrupts[i].reg);
 }
 return 0;
}
EXPORT_SYMBOL(legacy_gasket_interrupt_setup);
static int gasket_interrupt_setup(struct gasket_dev *gasket_dev)
{
 int i;
 struct gasket_interrupt_data *interrupt_data =
  gasket_dev->interrupt_data;
 if (!interrupt_data) {
  gasket_log_error(
   gasket_dev, "Interrupt data is not initialized.");
  return -EINVAL;
 }
 if (gasket_interrupt_is_legacy(interrupt_data)) {
  gasket_log_error(gasket_dev,
   "Unable to run interrupt setup on legacy device.");
  return -EPERM;
 }
 gasket_log_debug(gasket_dev, "Running interrupt setup.");
 for (i = 0; i < interrupt_data->num_interrupts; i++)
  interrupt_data->configured_registers[i] = false;
 for (i = 0; i < interrupt_data->num_msix_interrupts; i++)
  interrupt_data->mapped_registers[i] = 0;
 return 0;
}
void gasket_interrupt_cleanup(struct gasket_dev *gasket_dev)
{
 struct gasket_interrupt_data *interrupt_data =
  gasket_dev->interrupt_data;
 ulong flags;
 int i;
 if (!interrupt_data)
  return;
 gasket_interrupt_clear_all_eventfds(gasket_dev);
 gasket_interrupt_msix_teardown(interrupt_data);
 for (i = 0; i < interrupt_data->num_msix_interrupts; i++)
  kfree(interrupt_data->names[i]);
 kfree(interrupt_data->names);
 kfree(interrupt_data->configured_interrupts);
 kfree(interrupt_data->interrupt_counts);
 write_lock_irqsave(&interrupt_data->eventfd_ctx_lock, flags);
 kfree(interrupt_data->eventfd_ctxs);
 write_unlock_irqrestore(&interrupt_data->eventfd_ctx_lock, flags);
 kfree(interrupt_data->msix_entries);
}
int gasket_interrupt_system_status(struct gasket_dev *gasket_dev)
{
 if (!gasket_dev->interrupt_data) {
  gasket_log_info(gasket_dev, "Interrupt data is null.");
  return GASKET_STATUS_DEAD;
 }
 if (!gasket_dev->interrupt_data->msix_configured) {
  gasket_log_info(gasket_dev, "MSIx is not configured.");
  return GASKET_STATUS_LAMED;
 }
 if (gasket_interrupt_is_legacy(gasket_dev->interrupt_data)) {
  if (gasket_dev->interrupt_data->num_configured !=
   gasket_dev->interrupt_data->num_interrupts) {
   gasket_log_info(gasket_dev,
    "Not all interrupts were configured.");
   return GASKET_STATUS_LAMED;
  }
 }
 return GASKET_STATUS_ALIVE;
}
static int gasket_interrupt_set_eventfd(
 struct gasket_filp_data *filp_data, int interrupt, int event_fd)
{
 struct gasket_dev *gasket_dev = filp_data->gasket_dev;
 struct gasket_interrupt_data *interrupt_data =
  gasket_dev->interrupt_data;
 struct eventfd_ctx *ctx;
 ulong flags;
 if (gasket_dev->driver_desc->interrupt_permissions_cb) {
  int retval = gasket_dev->driver_desc->interrupt_permissions_cb(
   filp_data, interrupt);
  if (retval < 0)
   return retval;
 }
 ctx = eventfd_ctx_fdget(event_fd);
 if (IS_ERR(ctx))
  return PTR_ERR(ctx);
 write_lock_irqsave(&interrupt_data->eventfd_ctx_lock, flags);
 if (interrupt_data->eventfd_ctxs[interrupt] != NULL)
  eventfd_ctx_put(interrupt_data->eventfd_ctxs[interrupt]);
 interrupt_data->eventfd_ctxs[interrupt] = ctx;
 write_unlock_irqrestore(&interrupt_data->eventfd_ctx_lock, flags);
 return 0;
}
int legacy_gasket_interrupt_set_eventfd(
 struct gasket_filp_data *filp_data, int interrupt, int event_fd)
{
 struct gasket_dev *gasket_dev = filp_data->gasket_dev;
 struct gasket_interrupt_data *interrupt_data =
  gasket_dev->interrupt_data;
 if (!gasket_interrupt_is_legacy(interrupt_data)) {
  gasket_nodev_error(
   "Unable to run legacy set_eventfd on device.");
  return -EPERM;
 }
 if (interrupt < 0 || interrupt >= interrupt_data->num_interrupts) {
  gasket_nodev_error(
   "Unable to set eventfd on invalid interrupt number.");
  return -EINVAL;
 }
 return gasket_interrupt_set_eventfd(filp_data, interrupt, event_fd);
}
int legacy_gasket_interrupt_clear_eventfd(
 struct gasket_filp_data *filp_data, int interrupt)
{
 struct gasket_dev *gasket_dev = filp_data->gasket_dev;
 struct gasket_interrupt_data *interrupt_data =
  gasket_dev->interrupt_data;
 if (!gasket_interrupt_is_legacy(interrupt_data)) {
  gasket_nodev_error(
   "Unable to run legacy clear_eventfd on device.");
  return -EPERM;
 }
 if (interrupt < 0 || interrupt >= interrupt_data->num_interrupts) {
  gasket_nodev_error(
   "Trying to clear eventfd from invalid interrupt number.");
  return -EINVAL;
 }
 return gasket_interrupt_clear_eventfd(gasket_dev, filp_data, interrupt);
}
struct eventfd_ctx **gasket_interrupt_get_eventfd_ctxs(
 struct gasket_interrupt_data *interrupt_data)
{
 return interrupt_data->eventfd_ctxs;
}
EXPORT_SYMBOL(gasket_interrupt_get_eventfd_ctxs);
static ssize_t interrupt_sysfs_show(
 struct device *device, struct device_attribute *attr, char *buf)
{
 int i, ret;
 ssize_t written = 0, total_written = 0;
 struct gasket_interrupt_data *interrupt_data;
 struct gasket_dev *gasket_dev;
 struct gasket_sysfs_attribute *gasket_attr;
 enum interrupt_sysfs_attribute_type sysfs_type;
 gasket_dev = gasket_sysfs_get_device_data(device);
 if (gasket_dev == NULL)
  return 0;
 gasket_attr = gasket_sysfs_get_attr(device, attr);
 if (gasket_attr == NULL)
  return 0;
 sysfs_type = (enum interrupt_sysfs_attribute_type)
        gasket_attr->data.attr_type;
 interrupt_data = gasket_dev->interrupt_data;
 switch (sysfs_type) {
 case ATTR_INTERRUPT_COUNTS:
  for (i = 0; i < interrupt_data->num_msix_interrupts; ++i) {
   written = scnprintf(buf, PAGE_SIZE - total_written,
    "0x%02x: %ld\n", i,
    interrupt_data->interrupt_counts[i]);
   total_written += written;
   buf += written;
  }
  ret = total_written;
  break;
 default:
  gasket_log_error(
   gasket_dev, "Unknown attribute: %s", attr->attr.name);
  ret = 0;
  break;
 }
 return ret;
}
static irqreturn_t gasket_interrupt_handler(int irq, void *dev_id)
{
 struct eventfd_ctx *ctx;
 struct gasket_interrupt_data *interrupt_data = dev_id;
 ulong flags;
 bool valid_irq = false;
 int interrupt = -1;
 int i;
 for (i = 0; i < interrupt_data->num_msix_interrupts; i++) {
  if (interrupt_data->msix_entries[i].vector == irq) {
   valid_irq = true;
   interrupt = interrupt_data->msix_entries[i].entry;
   break;
  }
 }
 if (!valid_irq) {
  gasket_nodev_error("Received unknown irq %d", irq);
  return IRQ_HANDLED;
 }
 if (interrupt < 0 ||
  interrupt >= interrupt_data->num_msix_interrupts) {
  gasket_nodev_error(
   "Received irq %d for unconfigured interrupt %d",
   irq, interrupt);
  return IRQ_HANDLED;
 }
 trace_gasket_interrupt_event(
  interrupt_data->names[interrupt], interrupt);
 read_lock_irqsave(&interrupt_data->eventfd_ctx_lock, flags);
 ctx = interrupt_data->eventfd_ctxs[interrupt];
 if (ctx)
  eventfd_signal(ctx, 1);
 read_unlock_irqrestore(&interrupt_data->eventfd_ctx_lock, flags);
 ++(interrupt_data->interrupt_counts[interrupt]);
 return IRQ_HANDLED;
}
int gasket_interrupt_register_mapping(
 struct gasket_filp_data *filp_data,
 int interrupt, int event_fd, int bar_index, u64 reg)
{
 struct gasket_dev *gasket_dev = filp_data->gasket_dev;
 struct gasket_interrupt_data *interrupt_data =
  gasket_dev->interrupt_data;
 const struct gasket_interrupt_desc *interrupt_desc;
 int i;
 bool valid;
 int ret;
 int register_index;
 if (gasket_interrupt_is_legacy(interrupt_data)) {
  gasket_log_error(gasket_dev,
   "Unable to register_interrupt_mapping on legacy device.");
  return -EPERM;
 }
 if (interrupt < 0 || interrupt >= interrupt_data->num_msix_interrupts) {
  gasket_log_error(gasket_dev,
   "Interrupt number %d is invalid.", interrupt);
  return -EINVAL;
 }
 valid = false;
 for (i = 0; i < interrupt_data->num_interrupts; i++) {
  interrupt_desc = &interrupt_data->interrupts[i];
  if (interrupt_desc->bar_index == bar_index &&
   interrupt_desc->reg == reg) {
   valid = true;
   register_index = i;
   break;
  }
 }
 if (!valid) {
  gasket_log_error(gasket_dev,
   "Bar %d offset 0x%llx is not a valid interrupt register.",
   bar_index, reg);
  return -EINVAL;
 }
 if (interrupt_data->configured_registers[register_index]) {
  gasket_log_error(gasket_dev,
   "Bar %d offset 0x%llx at index %d is already configured.",
   bar_index, reg, register_index);
  return -EINVAL;
 }
 if (interrupt_data->configured_interrupts[interrupt]) {
  gasket_log_error(gasket_dev,
   "Interrupt %d has already been mapped.", interrupt);
  return -EINVAL;
 }
 ret = gasket_interrupt_set_eventfd(filp_data, interrupt, event_fd);
 if (ret) {
  gasket_log_error(gasket_dev,
   "Error when setting eventfd %d for interrupt %d: %d",
   event_fd, interrupt, ret);
  return ret;
 }
 ret = gasket_interrupt_configure(gasket_dev, interrupt);
 if (ret) {
  gasket_log_error(gasket_dev,
   "Error when configuring interrupt %d: %d",
   interrupt, ret);
  goto fail_interrupt_configure;
 }
 gasket_dev_write_32(gasket_dev, interrupt, bar_index, reg);
 interrupt_data->mapped_registers[interrupt] = register_index;
 interrupt_data->configured_registers[register_index] = true;
 return 0;
fail_interrupt_configure:
 gasket_interrupt_clear_eventfd(gasket_dev, filp_data, interrupt);
 return ret;
}
int gasket_interrupt_unregister_mapping(
 struct gasket_filp_data *filp_data, int interrupt)
{
 struct gasket_dev *gasket_dev = filp_data->gasket_dev;
 struct gasket_interrupt_data *interrupt_data;
 int register_index;
 interrupt_data = gasket_dev->interrupt_data;
 if (gasket_interrupt_is_legacy(interrupt_data)) {
  gasket_log_error(gasket_dev,
   "Unable to unregister interrupt on legacy device.");
  return -EPERM;
 }
 if (interrupt < 0 || interrupt >= interrupt_data->num_msix_interrupts) {
  gasket_log_error(gasket_dev,
   "Trying to unregister interrupt number %d ",
   interrupt);
  return -EINVAL;
 }
 register_index = interrupt_data->mapped_registers[interrupt];
 interrupt_data->configured_registers[register_index] = false;
 interrupt_data->mapped_registers[interrupt] = 0;
 gasket_interrupt_free_irq(interrupt_data, interrupt);
 return gasket_interrupt_clear_eventfd(gasket_dev, filp_data, interrupt);
}
EXPORT_SYMBOL(gasket_interrupt_unregister_mapping);
