/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2020 Google LLC.
 */
#ifndef __ACCEL_LIB_H__
#define __ACCEL_LIB_H__ 
#include <linux/accel.h>
#include <linux/accel_ioctl.h>
#include <linux/anon_inodes.h>
#include <linux/device.h>
#include <linux/errno.h>
#include <linux/kernel.h>
#include <linux/kobject.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/types.h>
#include <linux/uaccess.h>
#include <linux/uio_driver.h>
struct accel_dimm {
 s64 speed_hz;
 s64 size;
 s64 max_size;
 const char *type;
 struct kobject kobj;
};
static inline struct accel_dimm *to_accel_dimm(struct kobject *kobj)
{
 return container_of(kobj, struct accel_dimm, kobj);
}
extern struct kset *accel_dimm_create_kset(struct accel_dev *adev);
extern void accel_dimm_destroy_kset(struct kset *dimms);
extern int accel_dimm_add(struct kset *dimms, int id, struct accel_dimm *dimm);
extern long accel_lib_ioctl_version(unsigned long arg);
static inline int accel_lib_validate_ioctl(unsigned int cmd, unsigned long arg)
{
 void __user *p = (void __user *)arg;
 if (_IOC_TYPE(cmd) != ACCEL_IOCTL)
  return -ENOTTY;
 if (_IOC_DIR(cmd) & _IOC_READ) {
  if (!access_ok(p, _IOC_SIZE(cmd)))
   return -EFAULT;
 }
 if (_IOC_DIR(cmd) & _IOC_WRITE) {
  if (!access_ok(p, _IOC_SIZE(cmd)))
   return -EFAULT;
 }
 return 0;
}
struct accel_lib_user_regs_region {
 const int region_id;
 const char *region_name;
 const unsigned int base_BAR;
 const unsigned long BAR_offset;
 const unsigned long size;
};
#define PCI_MAX_NR_BARS (6)
extern int accel_lib_map_user_regs(unsigned long arg,
  struct accel_dev *adev, bool is_virtual_dev,
  struct accel_lib_user_regs_region regions[], int num_regions,
  phys_addr_t BAR_phys_addr[PCI_MAX_NR_BARS],
  void *vmalloc_base[PCI_MAX_NR_BARS]);
#endif
