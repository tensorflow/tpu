/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include "gasket_core.h"
#include "gasket_interrupt.h"
#include "gasket_ioctl.h"
#include "gasket_logging.h"
#include "gasket_page_table.h"
#include "gasket_sysfs.h"
#include <linux/delay.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/iommu.h>
#include <linux/sched.h>
#define CREATE_TRACE_POINTS 
#include <trace/events/gasket_mmap.h>
struct gasket_unforkable_mapping {
 int unforkable_map_count;
 pid_t last_mapped_unforkable_pid;
 char last_mapped_unforkable_process_name[TASK_COMM_LEN];
};
struct gasket_internal_desc {
  const struct gasket_device_desc *device_desc;
  struct mutex mutex;
  struct class *legacy_class;
  struct pci_driver pci;
  struct gasket_dev *devs[GASKET_DEV_MAX];
  struct gasket_unforkable_mapping unforkable_maps;
};
enum do_map_region_status {
 DO_MAP_REGION_SUCCESS,
 DO_MAP_REGION_FAILURE,
 DO_MAP_REGION_INVALID,
};
static int __init gasket_init(void);
static void gasket_exit(void);
static int gasket_pci_probe(
 struct pci_dev *pci_dev, const struct pci_device_id *id);
static void gasket_pci_remove(struct pci_dev *pci_dev);
static int gasket_setup_pci(struct pci_dev *pci_dev, struct gasket_dev *dev);
static void gasket_cleanup_pci(struct gasket_dev *dev);
static int gasket_map_pci_bar(struct gasket_dev *dev, int bar_num);
static void gasket_unmap_pci_bar(struct gasket_dev *dev, int bar_num);
static int gasket_alloc_dev(
 struct gasket_internal_desc *internal_desc, struct pci_dev *pci_dev,
 struct gasket_dev **pdev, const char *kobj_name);
static void gasket_free_dev(struct accel_dev *dev);
static int gasket_find_dev_slot(
 struct gasket_internal_desc *internal_desc, const char *kobj_name);
static int gasket_add_cdev(struct gasket_dev *gasket_dev,
 const struct file_operations *file_ops, struct module *owner);
static int gasket_enable_dev(struct gasket_dev *gasket_dev);
static void gasket_disable_dev(struct gasket_dev *gasket_dev);
static struct gasket_internal_desc *lookup_internal_desc(
 struct pci_dev *pci_dev);
static int gasket_sysfs_start(struct gasket_dev *dev);
static int gasket_sysfs_stop(struct gasket_dev *dev);
static ssize_t gasket_sysfs_data_show(
 struct device *device, struct device_attribute *attr, char *buf);
static int gasket_mmap(struct file *filp, struct vm_area_struct *vma);
static int gasket_open(struct inode *inode, struct file *file);
static int gasket_release(struct inode *inode, struct file *file);
static long gasket_ioctl(struct file *filp, uint cmd, ulong arg);
static bool gasket_mm_get_mapping_addrs(
 const struct gasket_mappable_region *region, ulong bar_offset,
 ulong requested_length, struct gasket_mappable_region *mappable_region,
 ulong *virt_offset);
static int gasket_get_hw_status(struct gasket_dev *gasket_dev);
static struct gasket_dev *gasket_dev_from_devt(dev_t devt);
static DEFINE_MUTEX(g_mutex);
static struct gasket_internal_desc g_descs[GASKET_FRAMEWORK_DESC_MAX];
static const struct gasket_num_name gasket_status_name_table[] = {
 { GASKET_STATUS_DEAD, "DEAD" },
 { GASKET_STATUS_ALIVE, "ALIVE" },
 { GASKET_STATUS_LAMED, "LAMED" },
 { GASKET_STATUS_DRIVER_EXIT, "DRIVER_EXITING" },
 { 0, NULL }
};
static DEFINE_HASHTABLE(cdev_to_gasket_dev, 10);
enum gasket_sysfs_attribute_type {
 ATTR_BAR_OFFSETS,
 ATTR_BAR_SIZES,
 ATTR_DRIVER_VERSION,
 ATTR_FRAMEWORK_VERSION,
 ATTR_DEVICE_TYPE,
 ATTR_HARDWARE_REVISION,
 ATTR_PCI_ADDRESS,
 ATTR_STATUS,
 ATTR_IS_DEVICE_OWNED,
 ATTR_DEVICE_OWNER,
 ATTR_WRITE_OPEN_COUNT,
 ATTR_RESET_COUNT,
 ATTR_USER_MEM_RANGES,
 ATTR_UNFORKABLE_MAP_COUNT,
 ATTR_LAST_UNFORKABLE_MAP_PID,
 ATTR_LAST_UNFORKABLE_MAP_NAME
};
static const struct file_operations gasket_file_ops = {
 .owner = THIS_MODULE,
 .llseek = no_llseek,
 .mmap = gasket_mmap,
 .open = gasket_open,
 .release = gasket_release,
 .unlocked_ioctl = gasket_ioctl,
};
static const struct gasket_sysfs_attribute gasket_sysfs_generic_attrs[] = {
 GASKET_SYSFS_RO(bar_offsets, gasket_sysfs_data_show, ATTR_BAR_OFFSETS),
 GASKET_SYSFS_RO(bar_sizes, gasket_sysfs_data_show, ATTR_BAR_SIZES),
 GASKET_SYSFS_RO(
  driver_version, gasket_sysfs_data_show, ATTR_DRIVER_VERSION),
 GASKET_SYSFS_RO(framework_version, gasket_sysfs_data_show,
  ATTR_FRAMEWORK_VERSION),
 GASKET_SYSFS_RO(device_type, gasket_sysfs_data_show, ATTR_DEVICE_TYPE),
 GASKET_SYSFS_RO(revision, gasket_sysfs_data_show,
  ATTR_HARDWARE_REVISION),
 GASKET_SYSFS_RO(pci_address, gasket_sysfs_data_show, ATTR_PCI_ADDRESS),
 GASKET_SYSFS_RO(status, gasket_sysfs_data_show, ATTR_STATUS),
 GASKET_SYSFS_RO(
  is_device_owned, gasket_sysfs_data_show, ATTR_IS_DEVICE_OWNED),
 GASKET_SYSFS_RO(device_owner, gasket_sysfs_data_show,
  ATTR_DEVICE_OWNER),
 GASKET_SYSFS_RO(write_open_count, gasket_sysfs_data_show,
  ATTR_WRITE_OPEN_COUNT),
 GASKET_SYSFS_RO(reset_count, gasket_sysfs_data_show, ATTR_RESET_COUNT),
 GASKET_SYSFS_RO(
  user_mem_ranges, gasket_sysfs_data_show, ATTR_USER_MEM_RANGES),
 GASKET_SYSFS_RO(
  unforkable_map_count, gasket_sysfs_data_show,
  ATTR_UNFORKABLE_MAP_COUNT),
 GASKET_SYSFS_RO(
  last_unforkable_mapped_page_pid, gasket_sysfs_data_show,
  ATTR_LAST_UNFORKABLE_MAP_PID),
 GASKET_SYSFS_RO(
  last_unforkable_mapped_page_process_name,
  gasket_sysfs_data_show,
  ATTR_LAST_UNFORKABLE_MAP_NAME),
 GASKET_END_OF_ATTR_ARRAY
};
MODULE_DESCRIPTION("Gasket driver framework");
MODULE_VERSION(GASKET_FRAMEWORK_VERSION);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Googler <noreply@google.com>");
module_init(gasket_init);
module_exit(gasket_exit);
static inline int check_and_invoke_callback(
 struct gasket_dev *gasket_dev, int (*cb_function)(struct gasket_dev *))
{
 int ret = 0;
 if (cb_function) {
  gasket_log_debug(
   gasket_dev, "Invoking device-specific callback.");
  mutex_lock(&gasket_dev->mutex);
  ret = cb_function(gasket_dev);
  mutex_unlock(&gasket_dev->mutex);
 }
 return ret;
}
static inline int check_and_invoke_callback_nolock(
 struct gasket_dev *gasket_dev, int (*cb_function)(struct gasket_dev *))
{
 int ret = 0;
 if (cb_function) {
  gasket_log_debug(
   gasket_dev, "Invoking device-specific callback.");
  ret = cb_function(gasket_dev);
 }
 return ret;
}
static int __init gasket_init(void)
{
 int i;
 gasket_nodev_info("Performing one-time init of the Gasket framework.");
 mutex_lock(&g_mutex);
 for (i = 0; i < GASKET_FRAMEWORK_DESC_MAX; i++) {
   g_descs[i].device_desc = NULL;
   mutex_init(&g_descs[i].mutex);
 }
 gasket_sysfs_init();
 mutex_unlock(&g_mutex);
 return 0;
}
static void gasket_exit(void)
{
}
int __gasket_register_device(const struct gasket_device_desc *device_desc,
                             const char *driver_name) {
  int i, ret;
  int desc_idx = -1;
  struct gasket_internal_desc *internal;
  gasket_nodev_info("Initializing Gasket framework device");
  mutex_lock(&g_mutex);
  for (i = 0; i < GASKET_FRAMEWORK_DESC_MAX; i++) {
    if (g_descs[i].device_desc == device_desc) {
      gasket_nodev_error("%s driver already loaded/registered",
                         device_desc->name);
      mutex_unlock(&g_mutex);
      return -EBUSY;
    }
  }
  for (i = 0; i < GASKET_FRAMEWORK_DESC_MAX; i++) {
    if (!g_descs[i].device_desc) {
      g_descs[i].device_desc = device_desc;
      desc_idx = i;
      break;
    }
  }
  mutex_unlock(&g_mutex);
  gasket_nodev_info("loaded %s driver, framework version %s", device_desc->name,
                    GASKET_FRAMEWORK_VERSION);
  if (desc_idx == -1) {
    gasket_nodev_error("too many Gasket drivers loaded: %d\n",
                       GASKET_FRAMEWORK_DESC_MAX);
    return -EBUSY;
  }
  gasket_nodev_info("Performing initial internal structure setup.");
  internal = &g_descs[desc_idx];
  mutex_init(&internal->mutex);
  memset(internal->devs, 0, sizeof(struct gasket_dev *) * GASKET_DEV_MAX);
  memset(&internal->pci, 0, sizeof(internal->pci));
  internal->pci.name = driver_name;
  internal->pci.id_table = device_desc->pci_id_table;
  internal->pci.probe = gasket_pci_probe;
  internal->pci.remove = gasket_pci_remove;
  internal->unforkable_maps.unforkable_map_count = 0;
  internal->unforkable_maps.last_mapped_unforkable_pid = -1;
  memset(internal->unforkable_maps.last_mapped_unforkable_process_name, 0,
         TASK_COMM_LEN);
  if (device_desc->legacy_support) {
    internal->legacy_class =
        class_create(device_desc->module, device_desc->name);
    if (IS_ERR_OR_NULL(internal->legacy_class)) {
      ret = PTR_ERR(internal->legacy_class);
      gasket_nodev_error("cannot register %s legacy class [ret=%d]",
                         device_desc->name, ret);
      goto fail1;
    }
  }
  gasket_nodev_info("Registering PCI driver.");
  ret = __pci_register_driver(&internal->pci, device_desc->module,
                              device_desc->name);
  if (ret) {
    gasket_nodev_error("cannot register pci driver [ret=%d]", ret);
    goto fail2;
  }
  if (device_desc->legacy_support) {
    gasket_nodev_info("Registering legacy driver.");
    ret = register_chrdev_region(
        MKDEV(device_desc->legacy_major, device_desc->legacy_minor),
        GASKET_DEV_MAX, device_desc->name);
    if (ret) {
      gasket_nodev_error("Cannot register char driver [ret=%d]", ret);
      goto fail3;
    }
  }
  gasket_nodev_info("Driver registered successfully.");
  return 0;
fail3:
 pci_unregister_driver(&internal->pci);
fail2:
  if (device_desc->legacy_support) class_destroy(internal->legacy_class);
fail1:
  g_descs[desc_idx].device_desc = NULL;
  return ret;
}
EXPORT_SYMBOL(__gasket_register_device);
void gasket_unregister_device(const struct gasket_device_desc *device_desc) {
  int i, desc_idx;
  struct gasket_internal_desc *internal_desc = NULL;
  mutex_lock(&g_mutex);
  for (i = 0; i < GASKET_FRAMEWORK_DESC_MAX; i++) {
    if (g_descs[i].device_desc == device_desc) {
      internal_desc = &g_descs[i];
      desc_idx = i;
      break;
    }
  }
  mutex_unlock(&g_mutex);
  if (internal_desc == NULL) {
    gasket_nodev_error("request to unregister unknown desc: %s",
                       device_desc->name);
    return;
  }
  pci_unregister_driver(&internal_desc->pci);
  if (device_desc->legacy_support) {
    unregister_chrdev_region(
        MKDEV(device_desc->legacy_major, device_desc->legacy_minor),
        GASKET_DEV_MAX);
    class_destroy(internal_desc->legacy_class);
  }
  for (i = 0; i < GASKET_DEV_MAX; i++) {
    mutex_lock(&g_mutex);
    while (g_descs[desc_idx].devs[i]) {
      mutex_unlock(&g_mutex);
      usleep_range(500, 1000);
      mutex_lock(&g_mutex);
    };
    mutex_unlock(&g_mutex);
  }
  g_descs[desc_idx].device_desc = NULL;
  gasket_nodev_info("removed %s driver", device_desc->name);
}
EXPORT_SYMBOL(gasket_unregister_device);
static int gasket_alloc_dev(
 struct gasket_internal_desc *internal_desc, struct pci_dev *pci_dev,
 struct gasket_dev **pdev, const char *kobj_name)
{
 int ret, dev_idx;
 const struct gasket_device_desc *device_desc = internal_desc->device_desc;
 const struct gasket_driver_desc *driver_desc;
 struct gasket_dev *gasket_dev;
 struct device *parent;
 gasket_nodev_info("Allocating a Gasket device.");
 if (IS_ERR_OR_NULL(pci_dev)) {
  gasket_nodev_error("PCI device is NULL!");
  return -ENODEV;
 }
 parent = &(pci_dev->dev);
 *pdev = NULL;
 dev_idx = gasket_find_dev_slot(internal_desc, kobj_name);
 if (dev_idx < 0)
  return dev_idx;
 if (internal_desc->device_desc->driver_desc) {
   driver_desc = internal_desc->device_desc->driver_desc;
 } else if (internal_desc->device_desc->driver_desc_cb) {
   driver_desc = internal_desc->device_desc->driver_desc_cb(pci_dev);
 } else {
   gasket_nodev_error("Must provide a driver desc or cb");
   return -EINVAL;
 }
 if (driver_desc == NULL) {
   gasket_nodev_info("Driver desc cb returned null");
   return 1;
 }
 gasket_dev = *pdev = kzalloc(sizeof(*gasket_dev), GFP_KERNEL);
 if (!gasket_dev) {
  gasket_nodev_error("no memory for device");
  return -ENOMEM;
 }
 internal_desc->devs[dev_idx] = gasket_dev;
 mutex_init(&gasket_dev->mutex);
 ret = accel_dev_init(&gasket_dev->accel_dev, parent, gasket_free_dev);
 if (ret) {
  gasket_nodev_error("Error initializing accel dev: %d", ret);
  internal_desc->devs[dev_idx] = NULL;
  return ret;
 }
 gasket_dev->accel_dev.accel_type = device_desc->name;
 gasket_dev->accel_dev.chip_vendor = "Google, LLC";
 gasket_dev->accel_dev.chip_model = driver_desc->chip_model;
 gasket_dev->accel_dev.chip_revision = driver_desc->chip_version;
 gasket_dev->accel_dev.logic_vendor = gasket_dev->accel_dev.chip_vendor;
 gasket_dev->accel_dev.logic_model = gasket_dev->accel_dev.chip_model;
 gasket_dev->accel_dev.logic_revision =
  gasket_dev->accel_dev.chip_revision;
 gasket_dev->accel_dev.fops = &gasket_file_ops;
 gasket_dev->internal_desc = internal_desc;
 gasket_dev->driver_desc = driver_desc;
 gasket_dev->pci_dev = pci_dev;
 gasket_dev->dev_idx = dev_idx;
 snprintf(gasket_dev->kobj_name, GASKET_NAME_MAX, "%s", kobj_name);
 gasket_dev->num_page_tables = driver_desc->num_page_tables;
 if (device_desc->legacy_support) {
   snprintf(gasket_dev->legacy_cdev_name, GASKET_NAME_MAX, "%s_%u",
            device_desc->name, gasket_dev->dev_idx);
   gasket_dev->legacy_devt =
       MKDEV(device_desc->legacy_major,
             device_desc->legacy_minor + gasket_dev->dev_idx);
   gasket_dev->legacy_cdev_added = 0;
   gasket_dev->legacy_device = device_create(
       internal_desc->legacy_class, parent, gasket_dev->legacy_devt, gasket_dev,
       gasket_dev->legacy_cdev_name);
   if (gasket_dev->legacy_device == NULL) {
     internal_desc->devs[dev_idx] = NULL;
     accel_dev_put(&gasket_dev->accel_dev);
     return 1;
   }
 }
 return 0;
}
static void gasket_free_dev(struct accel_dev *accel_dev)
{
 struct gasket_dev *gasket_dev;
 struct gasket_internal_desc *internal_desc;
 gasket_dev = container_of(accel_dev, struct gasket_dev, accel_dev);
 if (gasket_dev == NULL) {
  gasket_nodev_error(
   "Unable to find Gasket device from Accel device!");
  return;
 }
 internal_desc = gasket_dev->internal_desc;
 if (internal_desc == NULL) {
  gasket_nodev_error(
   "Unable to find internal device descriptor!");
  return;
 }
 mutex_lock(&internal_desc->mutex);
 internal_desc->devs[gasket_dev->dev_idx] = NULL;
 mutex_unlock(&internal_desc->mutex);
 kfree(gasket_dev);
}
static int gasket_find_dev_slot(
 struct gasket_internal_desc *internal_desc, const char *kobj_name)
{
 int i;
 mutex_lock(&internal_desc->mutex);
 for (i = 0; i < GASKET_DEV_MAX; i++) {
  if (internal_desc->devs[i] &&
   strcmp(internal_desc->devs[i]->kobj_name, kobj_name) ==
    0) {
   gasket_nodev_error("Duplicate device %s", kobj_name);
   mutex_unlock(&internal_desc->mutex);
   return -EBUSY;
  }
 }
 for (i = 0; i < GASKET_DEV_MAX; i++) {
  if (!internal_desc->devs[i])
   break;
 }
 if (i == GASKET_DEV_MAX) {
  gasket_nodev_info(
   "Too many registered devices; max %d", GASKET_DEV_MAX);
  mutex_unlock(&internal_desc->mutex);
  return -EBUSY;
 }
 mutex_unlock(&internal_desc->mutex);
 return i;
}
static int gasket_pci_probe(
 struct pci_dev *pci_dev, const struct pci_device_id *id)
{
 int ret;
 const char *kobj_name = dev_name(&pci_dev->dev);
 struct gasket_internal_desc *internal_desc;
 struct gasket_dev *gasket_dev;
 const struct gasket_device_desc *device_desc;
 const struct gasket_driver_desc *driver_desc;
 gasket_nodev_info("add Gasket device %s", kobj_name);
 mutex_lock(&g_mutex);
 internal_desc = lookup_internal_desc(pci_dev);
 mutex_unlock(&g_mutex);
 if (!internal_desc) {
  gasket_nodev_info("PCI probe called for unknown driver type");
  return -ENODEV;
 }
 ret = gasket_alloc_dev(internal_desc, pci_dev, &gasket_dev, kobj_name);
 if (ret)
  return ret;
 device_desc = internal_desc->device_desc;
 driver_desc = gasket_dev->driver_desc;
 ret = check_and_invoke_callback(
  gasket_dev, driver_desc->add_dev_cb);
 if (ret) {
  gasket_log_error(gasket_dev, "Error in add device cb: %d", ret);
  goto fail1;
 }
 ret = gasket_setup_pci(pci_dev, gasket_dev);
 if (ret)
  goto fail2;
 ret = gasket_enable_dev(gasket_dev);
 if (ret) {
   gasket_log_error(gasket_dev, "cannot setup %s device", device_desc->name);
   goto fail3;
 }
 accel_dev_set_state(&gasket_dev->accel_dev, "available");
 return 0;
fail3:
 gasket_cleanup_pci(gasket_dev);
fail2:
 check_and_invoke_callback(gasket_dev, driver_desc->remove_dev_cb);
fail1:
 accel_dev_put(&gasket_dev->accel_dev);
 return ret;
}
static void gasket_pci_remove(struct pci_dev *pci_dev)
{
 int i;
 struct gasket_internal_desc *internal_desc;
 struct gasket_dev *gasket_dev = NULL;
 const struct gasket_device_desc *device_desc;
 const struct gasket_driver_desc *driver_desc;
 mutex_lock(&g_mutex);
 internal_desc = lookup_internal_desc(pci_dev);
 if (!internal_desc) {
  mutex_unlock(&g_mutex);
  return;
 }
 mutex_unlock(&g_mutex);
 mutex_lock(&internal_desc->mutex);
 for (i = 0; i < GASKET_DEV_MAX; i++) {
  if (internal_desc->devs[i] &&
   internal_desc->devs[i]->pci_dev == pci_dev) {
   gasket_dev = internal_desc->devs[i];
   break;
  }
 }
 mutex_unlock(&internal_desc->mutex);
 if (!gasket_dev)
  return;
 device_desc = internal_desc->device_desc;
 driver_desc = gasket_dev->driver_desc;
 mutex_lock(&gasket_dev->mutex);
 gasket_dev->status = GASKET_STATUS_DRIVER_EXIT;
 if (gasket_dev->ownership.write_open_count > 0) {
  gasket_log_info(gasket_dev,
   "Waiting for %d active handles to close.",
   gasket_dev->ownership.write_open_count);
 }
 while (gasket_dev->ownership.write_open_count > 0) {
  mutex_unlock(&gasket_dev->mutex);
  usleep_range(500, 1000);
  mutex_lock(&gasket_dev->mutex);
 }
 mutex_unlock(&gasket_dev->mutex);
 gasket_log_info(gasket_dev, "removing %s device %s", device_desc->name,
                 gasket_dev->kobj_name);
 check_and_invoke_callback(gasket_dev, driver_desc->sysfs_cleanup_cb);
 gasket_sysfs_remove_mapping(&gasket_dev->accel_dev.dev);
 gasket_disable_dev(gasket_dev);
 gasket_cleanup_pci(gasket_dev);
 check_and_invoke_callback_nolock(
  gasket_dev, driver_desc->remove_dev_cb);
 if (gasket_dev->internal_desc->device_desc->legacy_support)
   device_destroy(internal_desc->legacy_class, gasket_dev->legacy_devt);
 accel_dev_put(&gasket_dev->accel_dev);
}
bool gasket_pci_is_iommu_enabled(struct pci_dev *pdev)
{
 return false;
}
EXPORT_SYMBOL(gasket_pci_is_iommu_enabled);
static void gasket_setup_pci_iommu(struct gasket_dev *gasket_dev)
{
 struct pci_dev *pdev = gasket_dev->pci_dev;
 const struct gasket_driver_desc *driver_desc = gasket_dev->driver_desc;
 if (gasket_pci_is_iommu_enabled(pdev)) {
  gasket_log_info(gasket_dev,
   "IOMMU Mappings: Already enabled");
 } else if (driver_desc->iommu_mappings == GASKET_IOMMU_PREFER) {
#if 0
#else
  gasket_log_warn(gasket_dev,
   "IOMMU Mappings: Cannot enable");
#endif
 }
}
static int gasket_setup_pci(
 struct pci_dev *pci_dev, struct gasket_dev *gasket_dev)
{
 int i, mapped_bars, ret;
 gasket_dev->pci_dev = pci_dev;
 ret = pci_enable_device(pci_dev);
 if (ret) {
  gasket_log_error(gasket_dev, "cannot enable PCI device");
  return ret;
 }
 pci_set_master(pci_dev);
 for (i = 0; i < GASKET_NUM_BARS; i++) {
  ret = gasket_map_pci_bar(gasket_dev, i);
  if (ret) {
   mapped_bars = i;
   goto fail;
  }
 }
 gasket_setup_pci_iommu(gasket_dev);
 return 0;
fail:
 for (i = 0; i < mapped_bars; i++)
  gasket_unmap_pci_bar(gasket_dev, i);
 pci_disable_device(pci_dev);
 return -ENOMEM;
}
static void gasket_cleanup_pci(struct gasket_dev *gasket_dev)
{
 int i;
 for (i = 0; i < GASKET_NUM_BARS; i++)
  gasket_unmap_pci_bar(gasket_dev, i);
 pci_disable_device(gasket_dev->pci_dev);
}
static int gasket_map_pci_bar(struct gasket_dev *gasket_dev, int bar_num)
{
  const struct gasket_driver_desc *driver_desc = gasket_dev->driver_desc;
  ulong desc_bytes = driver_desc->bar_descriptions[bar_num].size;
  int ret;
  if (desc_bytes == 0) return 0;
  gasket_dev->bar_data[bar_num].phys_base =
      (ulong)pci_resource_start(gasket_dev->pci_dev, bar_num);
  if (!gasket_dev->bar_data[bar_num].phys_base) {
    gasket_log_error(gasket_dev, "Cannot get BAR%u base address", bar_num);
    return -EINVAL;
  }
 gasket_dev->bar_data[bar_num].length_bytes =
  (ulong)pci_resource_len(gasket_dev->pci_dev, bar_num);
 if (gasket_dev->bar_data[bar_num].length_bytes < desc_bytes) {
  gasket_log_error(gasket_dev,
   "PCI BAR %u space is too small: %lu; expected >= %lu",
   bar_num, gasket_dev->bar_data[bar_num].length_bytes,
   desc_bytes);
  return -ENOMEM;
 }
 if (!request_mem_region(gasket_dev->bar_data[bar_num].phys_base,
      gasket_dev->bar_data[bar_num].length_bytes,
      accel_dev_name(&gasket_dev->accel_dev))) {
  gasket_log_error(gasket_dev,
   "Cannot get BAR %d memory region %p",
   bar_num, &gasket_dev->pci_dev->resource[bar_num]);
  return -EINVAL;
 }
 gasket_dev->bar_data[bar_num].virt_base =
     ioremap(gasket_dev->bar_data[bar_num].phys_base,
             gasket_dev->bar_data[bar_num].length_bytes);
 if (!gasket_dev->bar_data[bar_num].virt_base) {
  gasket_log_error(gasket_dev,
   "Cannot remap BAR %d memory region %p",
   bar_num, &gasket_dev->pci_dev->resource[bar_num]);
  ret = -ENOMEM;
  goto fail;
 }
 dma_set_mask(&gasket_dev->pci_dev->dev, DMA_BIT_MASK(64));
 dma_set_coherent_mask(&gasket_dev->pci_dev->dev, DMA_BIT_MASK(64));
 return 0;
fail:
 iounmap(gasket_dev->bar_data[bar_num].virt_base);
 release_mem_region(gasket_dev->bar_data[bar_num].phys_base,
  gasket_dev->bar_data[bar_num].length_bytes);
 return ret;
}
static void gasket_unmap_pci_bar(struct gasket_dev *dev, int bar_num)
{
 ulong base, bytes;
 const struct gasket_driver_desc *driver_desc = dev->driver_desc;
 if (driver_desc->bar_descriptions[bar_num].size == 0 ||
  !dev->bar_data[bar_num].virt_base)
  return;
 iounmap(dev->bar_data[bar_num].virt_base);
 dev->bar_data[bar_num].virt_base = NULL;
 base = pci_resource_start(dev->pci_dev, bar_num);
 if (!base) {
  gasket_log_error(
   dev, "cannot get PCI BAR%u base address", bar_num);
  return;
 }
 bytes = pci_resource_len(dev->pci_dev, bar_num);
 release_mem_region(base, bytes);
}
static int gasket_add_cdev(struct gasket_dev *gasket_dev,
 const struct file_operations *file_ops, struct module *owner)
{
 int ret;
 cdev_init(&gasket_dev->legacy_cdev, file_ops);
 gasket_dev->legacy_cdev.owner = owner;
 ret = cdev_add(&gasket_dev->legacy_cdev, gasket_dev->legacy_devt, 1);
 if (ret) {
  gasket_log_error(gasket_dev,
   "Cannot add char device [ret=%d]", ret);
  return ret;
 }
 gasket_dev->legacy_cdev_added = 1;
 return 0;
}
void gasket_add_cdev_mapping(struct gasket_dev *gasket_dev, dev_t devt)
{
 hash_add(cdev_to_gasket_dev, &gasket_dev->hlist_node, devt);
}
void gasket_remove_cdev_mapping(struct gasket_dev *gasket_dev)
{
 hash_del(&gasket_dev->hlist_node);
}
static int gasket_enable_dev(
 struct gasket_dev *gasket_dev)
{
 int i, tbl_idx;
 int ret;
 const struct gasket_device_desc *device_desc =
     gasket_dev->internal_desc->device_desc;
 const struct gasket_driver_desc *driver_desc = gasket_dev->driver_desc;
 if (driver_desc->legacy_interrupts && driver_desc->interrupts) {
  gasket_log_error(gasket_dev,
   "Can not use both legacy and non-legacy interrupt interfaces");
  return -EINVAL;
 }
 ret = accel_dev_register(&gasket_dev->accel_dev);
 if (ret) {
  gasket_log_error(gasket_dev,
   "Failed to register device with accel: %d", ret);
  return ret;
 }
 gasket_dev->accel_registered = true;
 gasket_log_info(gasket_dev, "Bound to dev node %s",
   gasket_dev->accel_dev.dev.kobj.name);
 ret = gasket_sysfs_start(gasket_dev);
 if (ret)
  goto fail1;
 if (driver_desc->legacy_interrupts != NULL) {
   ret = legacy_gasket_interrupt_init(
       gasket_dev, device_desc->name, driver_desc->legacy_interrupts,
       driver_desc->num_interrupts, driver_desc->legacy_interrupt_pack_width,
       driver_desc->legacy_interrupt_bar_index);
 } else {
   ret = gasket_interrupt_init(
       gasket_dev, device_desc->name, driver_desc->interrupts,
       driver_desc->num_interrupts, driver_desc->num_msix_interrupts);
 }
 if (ret) {
  gasket_log_error(gasket_dev,
   "Critical failure allocating interrupts: %d", ret);
  goto fail2;
 }
 for (tbl_idx = 0; tbl_idx < driver_desc->num_page_tables; tbl_idx++) {
  gasket_log_debug(
   gasket_dev, "Initializing page table %d.", tbl_idx);
  ret = gasket_page_table_init(&gasket_dev->page_table[tbl_idx],
   &gasket_dev->bar_data
    [driver_desc->page_table_configs[tbl_idx]
      .bar_index],
   &driver_desc->page_table_configs[tbl_idx],
   gasket_dev);
  if (ret) {
   gasket_log_error(gasket_dev,
    "Couldn't init page table %d: %d",
    tbl_idx, ret);
   goto fail3;
  }
  gasket_page_table_reset(gasket_dev->page_table[tbl_idx]);
 }
 ret = check_and_invoke_callback(
  gasket_dev, driver_desc->hardware_revision_cb);
 if (ret < 0) {
  gasket_log_error(
   gasket_dev, "Error getting hardware revision: %d", ret);
  goto fail3;
 }
 gasket_dev->hardware_revision = ret;
 if (driver_desc->firmware_version_cb) {
   ret = driver_desc->firmware_version_cb(gasket_dev,
                                          &gasket_dev->accel_dev.fw_version[0],
                                          &gasket_dev->accel_dev.fw_version[1],
                                          &gasket_dev->accel_dev.fw_version[2],
                                          &gasket_dev->accel_dev.fw_version[3]);
   if (ret) {
     gasket_log_error(gasket_dev, "Error getting firmware revision: %d", ret);
     goto fail3;
   }
 }
 ret = check_and_invoke_callback(
  gasket_dev, driver_desc->enable_dev_cb);
 if (ret) {
  gasket_log_error(
   gasket_dev, "Error in enable device cb: %d", ret);
  goto fail3;
 }
 gasket_dev->status = gasket_get_hw_status(gasket_dev);
 if (gasket_dev->status == GASKET_STATUS_DEAD)
  gasket_log_error(gasket_dev, "Device reported as unhealthy.");
 gasket_add_cdev_mapping(gasket_dev, gasket_dev->accel_dev.cdev.dev);
 if (gasket_dev->internal_desc->device_desc->legacy_support) {
   ret = gasket_add_cdev(gasket_dev, &gasket_file_ops,
                         gasket_dev->internal_desc->device_desc->module);
   if (ret) goto fail4;
   hash_add(cdev_to_gasket_dev, &gasket_dev->legacy_hlist_node,
            gasket_dev->legacy_devt);
 }
 return 0;
fail4:
 gasket_remove_cdev_mapping(gasket_dev);
 check_and_invoke_callback(gasket_dev, driver_desc->disable_dev_cb);
fail3:
 for (i = 0; i < tbl_idx; i++)
  gasket_page_table_cleanup(gasket_dev->page_table[i]);
 gasket_interrupt_cleanup(gasket_dev);
fail2:
 gasket_sysfs_stop(gasket_dev);
fail1:
 gasket_dev->accel_registered = false;
 accel_dev_unregister(&gasket_dev->accel_dev);
 return ret;
}
static void gasket_disable_dev(struct gasket_dev *gasket_dev)
{
  const struct gasket_driver_desc *driver_desc = gasket_dev->driver_desc;
  int i;
  for (i = 0; i < GASKET_MAX_CLONES; i++)
    if (gasket_dev->clones[i] != NULL)
      gasket_clone_cleanup(gasket_dev->clones[i]);
  gasket_dev->status = GASKET_STATUS_DEAD;
  gasket_remove_cdev_mapping(gasket_dev);
  gasket_sysfs_stop(gasket_dev);
  gasket_interrupt_cleanup(gasket_dev);
  for (i = 0; i < driver_desc->num_page_tables; ++i) {
    if (gasket_dev->page_table[i] != NULL) {
      gasket_page_table_reset(gasket_dev->page_table[i]);
      gasket_page_table_cleanup(gasket_dev->page_table[i]);
    }
  }
  if (gasket_dev->internal_desc->device_desc->legacy_support &&
      gasket_dev->legacy_cdev_added) {
    hash_del(&gasket_dev->legacy_hlist_node);
    cdev_del(&gasket_dev->legacy_cdev);
  }
 if (gasket_dev->accel_registered) {
  accel_dev_unregister(&gasket_dev->accel_dev);
  gasket_dev->accel_registered = false;
 }
 check_and_invoke_callback(gasket_dev, driver_desc->disable_dev_cb);
}
int gasket_sysfs_start(struct gasket_dev *gasket_dev)
{
  const struct gasket_driver_desc *driver_desc = gasket_dev->driver_desc;
  int ret;
  ret = gasket_sysfs_create_mapping(&gasket_dev->accel_dev.dev, gasket_dev);
  if (ret) return ret;
  ret = sysfs_create_link(&gasket_dev->accel_dev.dev.kobj,
                          &gasket_dev->pci_dev->dev.kobj,
                          dev_name(&gasket_dev->pci_dev->dev));
  if (ret) {
    gasket_log_error(gasket_dev, "Cannot create sysfs PCI link: %d", ret);
    goto fail1;
  }
  if (gasket_dev->internal_desc->device_desc->legacy_support) {
    ret = sysfs_create_link(&gasket_dev->legacy_device->kobj,
                            &gasket_dev->pci_dev->dev.kobj,
                            dev_name(&gasket_dev->pci_dev->dev));
    if (ret) {
      gasket_log_error(gasket_dev, "Cannot create legacy sysfs PCI link: %d",
                       ret);
      goto fail1;
    }
  }
 ret = gasket_sysfs_create_entries(
  &gasket_dev->accel_dev.dev, gasket_sysfs_generic_attrs);
 if (ret)
  goto fail1;
 ret = check_and_invoke_callback_nolock(
  gasket_dev, driver_desc->sysfs_setup_cb);
 if (ret) {
  gasket_log_error(
   gasket_dev, "Error in sysfs setup cb: %d", ret);
  goto fail2;
 }
 return 0;
fail2:
 check_and_invoke_callback(
  gasket_dev, driver_desc->sysfs_cleanup_cb);
fail1:
 gasket_sysfs_remove_mapping(&gasket_dev->accel_dev.dev);
 return ret;
}
static int gasket_sysfs_stop(struct gasket_dev *gasket_dev)
{
  const struct gasket_driver_desc *driver_desc = gasket_dev->driver_desc;
  gasket_sysfs_remove_mapping(&gasket_dev->accel_dev.dev);
  check_and_invoke_callback_nolock(gasket_dev, driver_desc->sysfs_cleanup_cb);
  return 0;
}
static void clone_release(struct accel_dev *dev)
{
}
int gasket_clone_create(struct gasket_dev *parent, struct gasket_dev *clone)
{
 int i, ret = 0;
 bool parent_had_space = false;
 const struct gasket_driver_desc *driver_desc = parent->driver_desc;
 __must_hold(&parent->mutex);
 if (parent->parent) {
  gasket_log_error(parent, "Clones cannot have clones!");
  return -EINVAL;
 }
 memset(clone, 0, sizeof(*clone));
 for (i = 0; i < GASKET_MAX_CLONES; i++)
  if (!parent->clones[i]) {
   parent_had_space = true;
   parent->clones[i] = clone;
   clone->clone_index = i;
   break;
  }
 if (!parent_had_space) {
  gasket_log_error(parent,
     "Parent already has the maximum number of clones (%d)",
     GASKET_MAX_CLONES);
  return -ENOMEM;
 }
 if (accel_dev_init(&clone->accel_dev, &parent->accel_dev.dev,
      &clone_release)) {
  gasket_log_error(parent, "Error initializing accel dev!");
  return -EIO;
 }
 clone->accel_dev.accel_type = parent->accel_dev.accel_type;
 clone->parent = parent;
 clone->internal_desc = parent->internal_desc;
 clone->driver_desc = driver_desc;
 clone->pci_dev = parent->pci_dev;
 clone->dev_idx = parent->dev_idx;
 memcpy(clone->kobj_name, parent->kobj_name, GASKET_NAME_MAX);
 memcpy(&clone->bar_data, &parent->bar_data, sizeof(clone->bar_data));
 clone->num_page_tables = parent->num_page_tables;
 for (i = 0; i < driver_desc->num_page_tables; i++)
  clone->page_table[i] = parent->page_table[i];
 clone->interrupt_data = parent->interrupt_data;
 clone->status = parent->status;
 clone->reset_count = parent->reset_count;
 clone->hardware_revision = parent->hardware_revision;
 mutex_init(&clone->mutex);
 clone->accel_dev.fops = &gasket_file_ops;
 ret = check_and_invoke_callback(
  clone, driver_desc->add_dev_cb);
 if (ret) {
  gasket_log_error(clone, "Error in add device cb: %d", ret);
  goto add_dev_fail;
 }
 if (accel_dev_register(&clone->accel_dev)) {
  gasket_log_error(clone, "Cannot accel-register clone of %d",
     parent->accel_dev.id);
  ret = -EIO;
  goto accel_register_fail;
 }
 if (gasket_sysfs_start(clone)) {
  ret = -EIO;
  goto sysfs_start_fail;
 }
 gasket_add_cdev_mapping(clone, clone->accel_dev.cdev.dev);
 return 0;
sysfs_start_fail:
 gasket_remove_cdev_mapping(clone);
 accel_dev_unregister(&clone->accel_dev);
 accel_dev_put(&clone->accel_dev);
accel_register_fail:
 check_and_invoke_callback(clone, driver_desc->remove_dev_cb);
add_dev_fail:
 for (i = 0; i < driver_desc->num_page_tables; i++)
  gasket_page_table_cleanup(clone->page_table[i]);
 for (i = 0; i < GASKET_MAX_CLONES; i++)
  if (parent->clones[i] == clone)
   parent->clones[i] = NULL;
 return ret;
}
EXPORT_SYMBOL(gasket_clone_create);
int gasket_clone_cleanup(struct gasket_dev *clone)
{
 int i;
 struct gasket_dev *parent = clone->parent;
 const struct gasket_driver_desc *driver_desc = parent->driver_desc;
 __must_hold(&parent->mutex);
 mutex_lock(&clone->mutex);
 gasket_remove_cdev_mapping(clone);
 gasket_sysfs_stop(clone);
 accel_dev_unregister(&clone->accel_dev);
 check_and_invoke_callback_nolock(clone, driver_desc->remove_dev_cb);
 for (i = 0; i < GASKET_MAX_CLONES; i++)
  if (parent->clones[i] == clone)
   parent->clones[i] = NULL;
 accel_dev_put(&clone->accel_dev);
 mutex_unlock(&clone->mutex);
 return 0;
}
EXPORT_SYMBOL(gasket_clone_cleanup);
static struct gasket_internal_desc *lookup_internal_desc(
 struct pci_dev *pci_dev)
{
 int i;
 __must_hold(&g_mutex);
 for (i = 0; i < GASKET_FRAMEWORK_DESC_MAX; i++) {
   if (g_descs[i].device_desc &&
       pci_match_id(g_descs[i].device_desc->pci_id_table, pci_dev))
     return &g_descs[i];
 }
 return NULL;
}
const char *gasket_num_name_lookup(
 uint num, const struct gasket_num_name *table)
{
 uint i = 0;
 while (table[i].snn_name) {
  if (num == table[i].snn_num)
   break;
  ++i;
 }
 return table[i].snn_name;
}
EXPORT_SYMBOL(gasket_num_name_lookup);
static int gasket_open(struct inode *inode, struct file *filp)
{
  int ret = 0;
  struct gasket_dev *gasket_dev;
  const struct gasket_driver_desc *driver_desc;
  struct gasket_ownership *ownership;
  struct gasket_filp_data *filp_data;
  char task_name[TASK_COMM_LEN];
  gasket_dev = gasket_dev_from_devt(inode->i_cdev->dev);
  if (!gasket_dev) {
    gasket_nodev_error("Unable to retrieve device data.");
    return -EINVAL;
  }
  driver_desc = gasket_dev->driver_desc;
  ownership = &gasket_dev->ownership;
  if (gasket_dev->status == GASKET_STATUS_DRIVER_EXIT) {
    gasket_log_info(gasket_dev,
                    "Cannot open device %s; driver is being removed.",
                    filp->f_path.dentry->d_iname);
    return -ENXIO;
  }
  filp->private_data = filp_data = kzalloc(sizeof(*filp_data), GFP_KERNEL);
  if (!filp_data) {
    gasket_log_error(gasket_dev, "Failed to allocate per-fd private data!");
    return -ENOMEM;
  }
  filp_data->gasket_dev = gasket_dev;
  get_task_comm(task_name, current);
  inode->i_size = 0;
  gasket_log_debug(gasket_dev,
                   "Attempting to open with tgid %u (%s) (f_mode: 0%03o, "
                   "fmode_write: %d is_root: %u)",
                   current->tgid, task_name, filp->f_mode,
                   (filp->f_mode & FMODE_WRITE), capable(CAP_SYS_ADMIN));
  if (!(filp->f_mode & FMODE_WRITE)) {
    gasket_log_debug(gasket_dev, "Allowing read-only opening.");
    return 0;
  }
 mutex_lock(&gasket_dev->mutex);
 gasket_log_debug(
  gasket_dev, "Current owner open count (owning tgid %u): %d.",
  ownership->owner, ownership->write_open_count);
 if (driver_desc->device_open_cb) {
   ret = driver_desc->device_open_cb(filp_data, filp);
   if (ret) {
     gasket_log_error(gasket_dev, "Error in device open cb: %d", ret);
     filp->private_data = NULL;
     kfree(filp_data);
     goto out;
   }
 }
 if (!ownership->is_owned) {
  ownership->is_owned = 1;
  ownership->owner = current->tgid;
  gasket_log_info(gasket_dev, "Device owner is now tgid %u",
   ownership->owner);
 }
 ownership->write_open_count++;
 gasket_log_debug(gasket_dev, "New open count (owning tgid %u): %d",
  ownership->owner, ownership->write_open_count);
 gasket_log_debug(
  gasket_dev, "Open of %s by tgid %u succeeded.",
  filp->f_path.dentry->d_iname, current->tgid);
out:
  mutex_unlock(&gasket_dev->mutex);
  return ret;
}
static void gasket_close(struct gasket_dev *gasket_dev,
                         struct gasket_filp_data *filp_data,
                         struct file *file) {
  int i;
  const struct gasket_driver_desc *driver_desc = gasket_dev->driver_desc;
  gasket_log_info(gasket_dev, "Device is now free");
  gasket_dev->ownership.is_owned = 0;
  gasket_dev->ownership.owner = 0;
  if (filp_data && driver_desc->device_close_cb)
    driver_desc->device_close_cb(filp_data, file);
  if (gasket_dev_is_overseer(gasket_dev)) return;
  for (i = 0; i < driver_desc->num_page_tables; ++i) {
    const struct gasket_page_table_config *config =
        &driver_desc->page_table_configs[i];
    bool do_unmap = true;
    if (config->owns_page_table_cb)
      do_unmap = config->owns_page_table_cb(gasket_dev, i);
    if (do_unmap) {
      gasket_page_table_unmap_all(gasket_dev->page_table[i]);
      gasket_page_table_garbage_collect(gasket_dev->page_table[i]);
    }
  }
}
static int gasket_release(struct inode *inode, struct file *file)
{
  struct gasket_filp_data *filp_data = file->private_data;
  struct gasket_dev *gasket_dev;
  struct gasket_ownership *ownership;
  const struct gasket_driver_desc *driver_desc;
  char task_name[TASK_COMM_LEN];
  gasket_dev = gasket_dev_from_devt(inode->i_cdev->dev);
  if (!gasket_dev) {
    gasket_nodev_error("Unable to retrieve device data");
    return -EINVAL;
  }
  driver_desc = gasket_dev->driver_desc;
  ownership = &gasket_dev->ownership;
  if (!filp_data || gasket_dev != filp_data->gasket_dev) {
    gasket_log_error(gasket_dev,
                     "Releasing device node that has missing/inconsistent "
                     "filp_data, not invoking callbacks");
    filp_data = NULL;
  }
 get_task_comm(task_name, current);
 mutex_lock(&gasket_dev->mutex);
 gasket_log_debug(gasket_dev,
  "Releasing device node. Call origin: tgid %u (%s) (f_mode: 0%03o, fmode_write: %d, is_root: %u)",
  current->tgid, task_name, file->f_mode,
  (file->f_mode & FMODE_WRITE), capable(CAP_SYS_ADMIN));
 gasket_log_debug(gasket_dev, "Current open count (owning tgid %u): %d",
  ownership->owner, ownership->write_open_count);
 if (file->f_mode & FMODE_WRITE) {
   if (filp_data && driver_desc->device_release_cb)
     driver_desc->device_release_cb(filp_data, file);
   if (ownership->write_open_count == 1)
     gasket_close(gasket_dev, filp_data, file);
   ownership->write_open_count--;
 }
 kfree(file->private_data);
 gasket_log_debug(gasket_dev, "New open count (owning tgid %u): %d",
  ownership->owner, ownership->write_open_count);
 mutex_unlock(&gasket_dev->mutex);
 return 0;
}
static int gasket_mmap_has_permissions(
 struct gasket_dev *gasket_dev, struct vm_area_struct *vma,
 int bar_permissions)
{
 int requested_permissions;
 if (capable(CAP_SYS_ADMIN))
  return 1;
 if (gasket_dev->status != GASKET_STATUS_ALIVE) {
  gasket_log_info(gasket_dev, "Device is dead.");
  return 0;
 }
 requested_permissions =
  (vma->vm_flags & (VM_WRITE | VM_READ | VM_EXEC));
 if (requested_permissions & ~(bar_permissions)) {
  gasket_log_info(gasket_dev,
   "Attempting to map a region with requested permissions 0x%x, but region has permissions 0x%x.",
   requested_permissions, bar_permissions);
  return 0;
 }
#ifndef STADIA_KERNEL
 if ((vma->vm_flags & VM_WRITE) &&
  (gasket_dev->ownership.is_owned &&
   gasket_dev->ownership.owner != current->tgid)) {
  gasket_log_info(gasket_dev,
   "Attempting to mmap a region for write without owning device.");
  return 0;
 }
#endif
 return 1;
}
int gasket_get_mmap_bar_index(const struct gasket_dev *gasket_dev,
                              ulong mmap_addr) {
  int i;
  const struct gasket_driver_desc *driver_desc;
  driver_desc = gasket_dev->driver_desc;
  for (i = 0; i < GASKET_NUM_BARS; ++i) {
    struct gasket_bar_desc bar_desc = driver_desc->bar_descriptions[i];
    if (bar_desc.permissions != GASKET_NOMAP) {
      if ((mmap_addr >= bar_desc.base) &&
          (mmap_addr < (bar_desc.base + bar_desc.size))) {
        return i;
      }
    }
  }
  return -EINVAL;
}
EXPORT_SYMBOL(gasket_get_mmap_bar_index);
static int gasket_get_phys_bar_index(
 const struct gasket_dev *gasket_dev, ulong phys_addr)
{
 int i;
 for (i = 0; i < GASKET_NUM_BARS; ++i) {
  const struct gasket_bar_data *bar_data =
   &gasket_dev->bar_data[i];
  if (phys_addr >= bar_data->phys_base &&
      phys_addr < bar_data->phys_base + bar_data->length_bytes)
   return i;
 }
 return -EINVAL;
}
static bool gasket_mm_get_mapping_addrs(
 const struct gasket_mappable_region *region, ulong bar_offset,
 ulong requested_length, struct gasket_mappable_region *mappable_region,
 ulong *virt_offset)
{
 ulong range_start = region->start;
 ulong range_length = region->length_bytes;
 ulong range_end = range_start + range_length;
 *virt_offset = 0;
 if (bar_offset + requested_length < range_start) {
  return false;
 } else if (bar_offset <= range_start) {
  mappable_region->start = range_start;
  *virt_offset = range_start - bar_offset;
  mappable_region->length_bytes =
   min(requested_length - *virt_offset, range_length);
  return (mappable_region->length_bytes != 0);
 } else if (bar_offset > range_start &&
     bar_offset < range_end) {
  mappable_region->start = bar_offset;
  *virt_offset = 0;
  mappable_region->length_bytes = min(
   requested_length, range_end - bar_offset);
  return (mappable_region->length_bytes != 0);
 }
 return false;
}
int gasket_mm_unmap_region(
 const struct gasket_dev *gasket_dev, struct vm_area_struct *vma,
 int map_bar_index,
 const struct gasket_mappable_region *map_region)
{
 u64 phys_addr;
 int bar_index;
 ulong bar_offset;
 ulong virt_offset;
 struct gasket_mappable_region mappable_region;
 if (vma->vm_private_data != gasket_dev)
  return -EINVAL;
 if (map_region->length_bytes == 0)
  return 0;
 phys_addr = vma->vm_pgoff << PAGE_SHIFT;
 bar_index = gasket_get_phys_bar_index(gasket_dev, phys_addr);
 if (bar_index < 0) {
  gasket_log_error(gasket_dev,
   "Unable to find matching bar for physical address %#llx",
   phys_addr);
  trace_gasket_mmap_exit(bar_index);
  return bar_index;
 }
 if (bar_index != map_bar_index) {
  gasket_log_debug(gasket_dev,
   "Found VMA for BAR%d, but looking to unmap from BAR%d",
   bar_index, map_bar_index);
  return 1;
 }
 bar_offset = phys_addr - gasket_dev->bar_data[bar_index].phys_base;
 if (!gasket_mm_get_mapping_addrs(map_region, bar_offset,
  vma->vm_end - vma->vm_start, &mappable_region, &virt_offset))
  return 1;
 zap_vma_ptes(vma, vma->vm_start + virt_offset,
  DIV_ROUND_UP(mappable_region.length_bytes, PAGE_SIZE) *
   PAGE_SIZE); return 0;
}
EXPORT_SYMBOL(gasket_mm_unmap_region);
static enum do_map_region_status do_map_region(
 struct gasket_dev *gasket_dev, struct vm_area_struct *vma,
 int bar_index, ulong bar_offset,
 const struct gasket_mappable_region *mappable_region)
{
 const ulong max_chunk_size = 64 * 1024 * 1024;
 ulong chunk_size, mapped_bytes = 0;
 ulong virt_offset;
 struct gasket_mappable_region region_to_map;
 ulong phys_offset, map_length;
 ulong virt_base, phys_base;
 int ret;
 if (!gasket_mm_get_mapping_addrs(mappable_region, bar_offset,
  vma->vm_end - vma->vm_start, &region_to_map, &virt_offset))
  return DO_MAP_REGION_INVALID;
 phys_offset = region_to_map.start;
 map_length = region_to_map.length_bytes;
 virt_base = vma->vm_start + virt_offset;
 phys_base = gasket_dev->bar_data[bar_index].phys_base + phys_offset;
 while (mapped_bytes < map_length) {
  chunk_size = min(max_chunk_size,
   map_length - mapped_bytes);
  cond_resched();
  ret = io_remap_pfn_range(vma,
   virt_base + mapped_bytes,
   (phys_base + mapped_bytes) >> PAGE_SHIFT,
   chunk_size, vma->vm_page_prot);
  if (ret) {
   gasket_log_error(gasket_dev,
    "Error remapping PFN range.");
   return DO_MAP_REGION_FAILURE;
  }
  mapped_bytes += chunk_size;
 }
 return DO_MAP_REGION_SUCCESS;
}
static int gasket_mmap(struct file *filp, struct vm_area_struct *vma)
{
 int i, ret;
 int bar_index;
 ulong bar_offset;
 int has_mapped_anything = 0;
 vm_flags_t req_perms;
 ulong raw_offset, vma_size;
 const struct gasket_driver_desc *driver_desc;
 struct gasket_filp_data *filp_data =
     (struct gasket_filp_data *)filp->private_data;
 struct gasket_dev *gasket_dev = filp_data->gasket_dev;
 const struct gasket_bar_desc *bar_desc;
 struct gasket_mappable_region *map_regions = NULL;
 int num_map_regions = 0;
 enum do_map_region_status map_status;
 if (!gasket_dev) {
  gasket_nodev_error("Unable to retrieve device data");
  trace_gasket_mmap_exit(-EINVAL);
  return -EINVAL;
 }
 driver_desc = gasket_dev->driver_desc;
 if (vma->vm_start & (PAGE_SIZE - 1)) {
  gasket_log_error(
   gasket_dev, "Base address not page-aligned: 0x%p\n",
   (void *)vma->vm_start);
  trace_gasket_mmap_exit(-EINVAL);
  return -EINVAL;
 }
 raw_offset = vma->vm_pgoff << PAGE_SHIFT;
 vma_size = vma->vm_end - vma->vm_start;
 trace_gasket_mmap_entry(
  dev_name(&gasket_dev->accel_dev.dev),
  raw_offset, vma_size);
 bar_index = gasket_get_mmap_bar_index(gasket_dev, raw_offset);
 if (bar_index < 0) {
  gasket_log_error(gasket_dev,
   "Unable to find matching bar for address 0x%lx",
   raw_offset);
  trace_gasket_mmap_exit(-EINVAL);
  return -EINVAL;
 }
 bar_desc = &driver_desc->bar_descriptions[bar_index];
 bar_offset = raw_offset - bar_desc->base;
 vma->vm_flags &= ~(VM_MAYREAD | VM_MAYWRITE | VM_MAYEXEC | VM_MAYSHARE);
 req_perms = vma->vm_flags & (VM_READ | VM_WRITE | VM_EXEC);
 if ((req_perms & bar_desc->permissions) != req_perms) {
  gasket_log_error(gasket_dev,
   "Invalid permissions for BAR%d. Want: %lu; has: %lu",
   bar_index, vma->vm_flags, bar_desc->permissions);
  return -EPERM;
 }
 if (driver_desc->get_mappable_regions_cb) {
   ret = driver_desc->get_mappable_regions_cb(filp_data, bar_index,
                                              &map_regions, &num_map_regions);
   if (ret) return ret;
 } else {
  if (!gasket_mmap_has_permissions(gasket_dev, vma,
       bar_desc->permissions)) {
   gasket_log_error(
    gasket_dev, "Permission checking failed.");
   trace_gasket_mmap_exit(-EPERM);
   return -EPERM;
  }
  num_map_regions = bar_desc->num_mappable_regions;
  map_regions = kcalloc(
   num_map_regions, sizeof(*bar_desc->mappable_regions),
   GFP_KERNEL);
  if (map_regions) {
   memcpy(map_regions, bar_desc->mappable_regions,
    num_map_regions *
     sizeof(*bar_desc->mappable_regions));
  }
 }
 if (map_regions == NULL || num_map_regions == 0) {
  gasket_log_info(gasket_dev, "No mappable regions returned!");
  return -EINVAL;
 }
 vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
 for (i = 0; i < num_map_regions; i++) {
  if ((req_perms & map_regions[i].flags) != req_perms) {
   gasket_log_info(gasket_dev,
      "Skipping mappable region 0x%lx; perm mismatch want: 0x%lx, have: 0x%lx",
      (ulong) map_regions[i].start,
      req_perms,
      map_regions[i].flags);
   continue;
  }
  map_status = do_map_region(
   gasket_dev, vma, bar_index, bar_offset,
   &map_regions[i]);
  if (map_status == DO_MAP_REGION_INVALID)
   continue;
  if (map_status == DO_MAP_REGION_FAILURE) {
   gasket_log_error(gasket_dev,
    "Error mapping region %d: 0x%llx/0x%llx",
    i, map_regions[i].start,
    map_regions[i].length_bytes);
   ret = -EINVAL;
   goto map_fail;
  }
  has_mapped_anything = 1;
 }
 if (!has_mapped_anything) {
  gasket_log_error(gasket_dev,
   "Map request did not contain a valid region.");
  ret = -EINVAL;
  goto exit;
 }
 vma->vm_private_data = gasket_dev;
 vma->vm_pgoff =
  (gasket_dev->bar_data[bar_index].phys_base >> PAGE_SHIFT) +
   vma->vm_pgoff - (bar_desc->base >> PAGE_SHIFT);
 ret = 0;
 goto exit;
map_fail:
 zap_vma_ptes(vma, vma->vm_start, vma_size);
exit:
 kfree(map_regions);
 trace_gasket_mmap_exit(ret);
 return ret;
}
void gasket_mapped_unforkable_page(struct gasket_dev *gasket_dev)
{
 struct gasket_internal_desc *internal_desc;
 struct gasket_unforkable_mapping *unforkable_maps;
 mutex_lock(&gasket_dev->internal_desc->mutex);
 internal_desc = gasket_dev->internal_desc;
 unforkable_maps = &internal_desc->unforkable_maps;
 unforkable_maps->unforkable_map_count++;
 unforkable_maps->last_mapped_unforkable_pid = current->pid;
 get_task_comm(unforkable_maps->last_mapped_unforkable_process_name,
  current);
 mutex_unlock(&gasket_dev->internal_desc->mutex);
}
EXPORT_SYMBOL(gasket_mapped_unforkable_page);
static int gasket_get_hw_status(struct gasket_dev *gasket_dev)
{
 int i;
 enum gasket_status status = GASKET_STATUS_ALIVE;
 const struct gasket_driver_desc *driver_desc = gasket_dev->driver_desc;
 if (driver_desc->device_status_cb) {
  gasket_log_debug(
   gasket_dev, "Invoking device-specific callback.");
  status = driver_desc->device_status_cb(gasket_dev);
 }
 if (status != GASKET_STATUS_ALIVE) {
  gasket_log_info(
   gasket_dev, "Hardware reported status %d.", status);
  return status;
 }
 status = gasket_interrupt_system_status(gasket_dev);
 if (status != GASKET_STATUS_ALIVE) {
  gasket_log_info(gasket_dev,
   "Interrupt system reported status %d.", status);
  return status;
 }
 for (i = 0; i < driver_desc->num_page_tables; ++i) {
  status = gasket_page_table_system_status(
   gasket_dev->page_table[i]);
  if (status != GASKET_STATUS_ALIVE) {
   gasket_log_info(
    gasket_dev, "Page table %d reported status %d.",
    i, status);
   return status;
  }
 }
 return GASKET_STATUS_ALIVE;
}
static long gasket_ioctl(struct file *filp, uint cmd, ulong arg)
{
  struct gasket_filp_data *filp_data;
  struct gasket_dev *gasket_dev;
  const struct gasket_driver_desc *driver_desc;
  char path[256];
  if (filp == NULL) return -ENODEV;
  filp_data = (struct gasket_filp_data *)filp->private_data;
  if (filp_data == NULL || filp_data->gasket_dev == NULL) {
    gasket_nodev_error("Unable to find Gasket structure for file %s",
                       d_path(&filp->f_path, path, 256));
    return -ENODEV;
  }
  gasket_dev = filp_data->gasket_dev;
  driver_desc = gasket_dev->driver_desc;
  if (driver_desc == NULL) {
    gasket_log_error(gasket_dev, "Unable to find device descriptor for file %s",
                     d_path(&filp->f_path, path, 256));
    return -ENODEV;
  }
 if (!gasket_is_supported_ioctl(cmd)) {
  if (driver_desc->ioctl_handler_cb)
   return driver_desc->ioctl_handler_cb(filp, cmd, arg);
  gasket_log_error(
   gasket_dev, "Received unknown ioctl 0x%x", cmd);
  return -EINVAL;
 }
 return gasket_handle_ioctl(filp, cmd, arg);
}
int gasket_reset(struct gasket_dev *gasket_dev, uint reset_type)
{
 int ret;
 if (gasket_dev->parent) mutex_lock(&gasket_dev->parent->mutex);
 mutex_lock(&gasket_dev->mutex);
 ret = gasket_reset_nolock(gasket_dev, reset_type);
 mutex_unlock(&gasket_dev->mutex);
 if (gasket_dev->parent) mutex_unlock(&gasket_dev->parent->mutex);
 return ret;
}
EXPORT_SYMBOL(gasket_reset);
int gasket_reset_nolock(struct gasket_dev *gasket_dev, uint reset_type)
{
 int ret;
 int i;
 const struct gasket_driver_desc *driver_desc;
 driver_desc = gasket_dev->driver_desc;
 if (!driver_desc->device_reset_cb) {
  gasket_log_error(
   gasket_dev, "No device reset callback was registered.");
  return -EINVAL;
 }
 ret = driver_desc->device_reset_cb(gasket_dev, reset_type);
 if (ret) {
  gasket_log_error(gasket_dev,
   "Device reset cb returned %d; aborting reset.", ret);
  return ret;
 }
 for (i = 0; i < driver_desc->num_page_tables; ++i)
  gasket_page_table_reset(gasket_dev->page_table[i]);
 ret = gasket_interrupt_reinit(gasket_dev);
 if (ret) {
  gasket_log_error(
   gasket_dev, "Unable to reinit interrupts: %d.", ret);
  return ret;
 }
 gasket_dev->status = gasket_get_hw_status(gasket_dev);
 if (gasket_dev->status == GASKET_STATUS_DEAD) {
  gasket_log_error(gasket_dev, "Device reported as dead.");
  return -EINVAL;
 }
 return 0;
}
EXPORT_SYMBOL(gasket_reset_nolock);
static ssize_t gasket_write_mappable_regions(
 char *buf, const struct gasket_driver_desc *driver_desc, int bar_index)
{
 int i;
 ssize_t written;
 ssize_t total_written = 0;
 ulong min_addr, max_addr;
 struct gasket_bar_desc bar_desc =
  driver_desc->bar_descriptions[bar_index];
 if (bar_desc.permissions == GASKET_NOMAP)
  return 0;
 for (i = 0; (i < bar_desc.num_mappable_regions) &&
  (total_written < PAGE_SIZE); i++) {
   min_addr = bar_desc.mappable_regions[i].start;
   max_addr = bar_desc.mappable_regions[i].start +
              bar_desc.mappable_regions[i].length_bytes;
   written = scnprintf(buf, PAGE_SIZE - total_written, "0x%08lx-0x%08lx\n",
                       min_addr, max_addr);
   total_written += written;
   buf += written;
 }
 return total_written;
}
static ssize_t gasket_sysfs_data_show(
 struct device *device, struct device_attribute *attr, char *buf)
{
 int i, ret = 0;
 ssize_t current_written;
 const struct gasket_driver_desc *driver_desc;
 struct gasket_dev *gasket_dev;
 struct gasket_sysfs_attribute *gasket_attr;
 const struct gasket_bar_desc *bar_desc;
 enum gasket_sysfs_attribute_type sysfs_type;
 const struct gasket_unforkable_mapping *unforkable_maps;
 gasket_dev = gasket_sysfs_get_device_data(device);
 if (gasket_dev == NULL)
  return 0;
 gasket_attr = gasket_sysfs_get_attr(device, attr);
 if (gasket_attr == NULL) {
  return 0;
 }
 driver_desc = gasket_dev->driver_desc;
 unforkable_maps = &gasket_dev->internal_desc->unforkable_maps;
 sysfs_type =
  (enum gasket_sysfs_attribute_type) gasket_attr->data.attr_type;
 switch (sysfs_type) {
 case ATTR_BAR_OFFSETS:
  for (i = 0; i < GASKET_NUM_BARS; i++) {
   bar_desc = &driver_desc->bar_descriptions[i];
   if (bar_desc->size == 0)
    continue;
   current_written = snprintf(buf, PAGE_SIZE - ret,
    "%d: 0x%lx\n", i, (ulong) bar_desc->base);
   buf += current_written;
   ret += current_written;
  }
  break;
 case ATTR_BAR_SIZES:
  for (i = 0; i < GASKET_NUM_BARS; i++) {
   bar_desc = &driver_desc->bar_descriptions[i];
   if (bar_desc->size == 0)
    continue;
   current_written = snprintf(buf, PAGE_SIZE - ret,
    "%d: 0x%lx\n", i, (ulong) bar_desc->size);
   buf += current_written;
   ret += current_written;
  }
  break;
 case ATTR_DRIVER_VERSION:
   ret = snprintf(buf, PAGE_SIZE, "%s\n",
                  gasket_dev->driver_desc->driver_version);
   break;
 case ATTR_FRAMEWORK_VERSION:
  ret = snprintf(
   buf, PAGE_SIZE, "%s\n", GASKET_FRAMEWORK_VERSION);
  break;
 case ATTR_DEVICE_TYPE:
   ret = snprintf(buf, PAGE_SIZE, "%s\n",
                  gasket_dev->internal_desc->device_desc->name);
   break;
 case ATTR_HARDWARE_REVISION:
  ret = snprintf(
   buf, PAGE_SIZE, "%d\n", gasket_dev->hardware_revision);
  break;
 case ATTR_PCI_ADDRESS:
  ret = snprintf(buf, PAGE_SIZE, "%s\n", gasket_dev->kobj_name);
  break;
 case ATTR_STATUS:
  ret = snprintf(buf, PAGE_SIZE, "%s\n",
   gasket_num_name_lookup(
    gasket_dev->status, gasket_status_name_table));
  break;
 case ATTR_IS_DEVICE_OWNED:
  ret = snprintf(buf, PAGE_SIZE, "%d\n",
   gasket_dev->ownership.is_owned);
  break;
 case ATTR_DEVICE_OWNER:
  ret = snprintf(buf, PAGE_SIZE, "%d\n",
   gasket_dev->ownership.owner);
  break;
 case ATTR_WRITE_OPEN_COUNT:
  ret = snprintf(buf, PAGE_SIZE, "%d\n",
   gasket_dev->ownership.write_open_count);
  break;
 case ATTR_RESET_COUNT:
  ret = snprintf(buf, PAGE_SIZE, "%d\n", gasket_dev->reset_count);
  break;
 case ATTR_USER_MEM_RANGES:
  for (i = 0; i < GASKET_NUM_BARS; ++i) {
   current_written = gasket_write_mappable_regions(
    buf, driver_desc, i);
   buf += current_written;
   ret += current_written;
  }
  break;
 case ATTR_UNFORKABLE_MAP_COUNT:
  ret = snprintf(buf, PAGE_SIZE, "%d\n",
   unforkable_maps->unforkable_map_count);
  break;
 case ATTR_LAST_UNFORKABLE_MAP_PID:
  ret = snprintf(buf, PAGE_SIZE, "%d\n",
   unforkable_maps->last_mapped_unforkable_pid);
  break;
 case ATTR_LAST_UNFORKABLE_MAP_NAME:
  ret = snprintf(buf, PAGE_SIZE, "%s\n",
   unforkable_maps->last_mapped_unforkable_process_name);
  break;
 default:
  gasket_log_error(
   gasket_dev, "Unknown attribute: %s", attr->attr.name);
  ret = 0;
  break;
 }
 return ret;
}
static struct gasket_dev *gasket_dev_from_devt(dev_t devt)
{
 struct gasket_dev *hash_entry;
 hash_for_each_possible(
  cdev_to_gasket_dev, hash_entry, hlist_node, devt) {
  if (hash_entry->accel_dev.cdev.dev == devt)
   return hash_entry;
 }
 hash_for_each_possible(
  cdev_to_gasket_dev, hash_entry, legacy_hlist_node, devt) {
   if (hash_entry->internal_desc->device_desc->legacy_support &&
       hash_entry->legacy_devt == devt) {
     return hash_entry;
   }
 }
 return NULL;
}
bool gasket_dev_is_overseer(struct gasket_dev *gasket_dev)
{
 int i;
 for (i = 0; i < GASKET_MAX_CLONES; i++)
  if (gasket_dev->clones[i])
   return true;
 return false;
}
EXPORT_SYMBOL(gasket_dev_is_overseer);
