/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include <linux/accel_lib.h>
#include <linux/sched.h>
#include <linux/vmalloc.h>
long accel_lib_ioctl_version(unsigned long arg)
{
 static const struct accel_ioctl_version version = {
  .major = ACCEL_VERSION_MAJOR,
  .minor = ACCEL_VERSION_MINOR,
  .patch = ACCEL_VERSION_PATCH,
 };
 if (__copy_to_user((struct accel_ioctl_version __user *)arg,
    &version, sizeof(version))) {
  return -EFAULT;
 }
 return 0;
}
EXPORT_SYMBOL(accel_lib_ioctl_version);
struct mapped_user_region {
 struct accel_dev *adev;
 resource_size_t size;
 bool is_virtual_dev;
 union {
  struct {
   phys_addr_t regs_paddr;
  } phys;
  struct {
   void *base;
   unsigned long pgoff;
  } virt;
 };
};
static int accel_lib_user_regs_fd_mmap(struct file *file,
  struct vm_area_struct *vma)
{
 struct mapped_user_region *mur = file->private_data;
 unsigned long off, physical_pfn;
 long vsize, psize;
 int ret;
 BUG_ON(!mur);
 off = vma->vm_pgoff << PAGE_SHIFT;
 vsize = vma->vm_end - vma->vm_start;
 psize = mur->size;
 dev_dbg(&(mur->adev->dev),
  "user_reg_mmap for (%lu bytes @ offset 0x%lX) by pid %d",
  vsize, off, current->tgid);
 if (off > psize || (off + vsize) > psize) {
  dev_dbg(&(mur->adev->dev),
   "user_reg_mmap beyond (%lu bytes @ offset 0x%lX) by pid %d?",
   vsize, off, current->tgid);
  return -EINVAL;
 }
 if (mur->is_virtual_dev) {
  ret = remap_vmalloc_range(vma, mur->virt.base,
   mur->virt.pgoff + vma->vm_pgoff);
 } else {
  physical_pfn = (mur->phys.regs_paddr >> PAGE_SHIFT) +
    vma->vm_pgoff;
  ret = remap_pfn_range(vma, vma->vm_start, physical_pfn,
    vsize, vma->vm_page_prot);
 }
 if (ret) {
  dev_dbg(&(mur->adev->dev),
   "user_reg_mmap: Error in remap function");
  return ret;
 }
 return 0;
}
static int accel_lib_user_regs_fd_release(struct inode *inode,
  struct file *file)
{
 struct mapped_user_region *mur = file->private_data;
 BUG_ON(!mur);
 dev_dbg(&(mur->adev->dev),
  "All references to user_regs fd given up by user\n");
 file->private_data = NULL;
 kfree(mur);
 return 0;
}
const struct file_operations accel_lib_user_regs_ctx_fd_fops = {
 .owner = THIS_MODULE,
 .mmap = accel_lib_user_regs_fd_mmap,
 .release = accel_lib_user_regs_fd_release,
};
static struct accel_lib_user_regs_region *find_region(
  struct accel_lib_user_regs_region *regions,
  int num_regions, int region_id)
{
 int i;
 for (i = 0; i < num_regions; i++) {
  if (regions[i].region_id == region_id)
   return &regions[i];
 }
 return NULL;
}
int accel_lib_map_user_regs(unsigned long arg,
  struct accel_dev *adev, bool is_virtual_dev,
  struct accel_lib_user_regs_region regions[], int num_regions,
  phys_addr_t BAR_phys_addr[PCI_MAX_NR_BARS],
  void *vmalloc_base[PCI_MAX_NR_BARS])
{
 struct accel_ioctl_map_user_regs ioctl_arg;
 struct accel_lib_user_regs_region *region;
 struct mapped_user_region *mur;
 int fd;
 char *tmp_inode_name;
 if (__copy_from_user(&ioctl_arg, (void __user *)arg,
        sizeof(ioctl_arg)))
  return -EFAULT;
 region = find_region(regions, num_regions, ioctl_arg.region_id);
 if (region == NULL)
  return -EINVAL;
 BUG_ON(region->base_BAR >= PCI_MAX_NR_BARS);
 ioctl_arg.region_size = region->size;
 if (__copy_to_user((void __user *)arg, &ioctl_arg, sizeof(ioctl_arg)))
  return -EFAULT;
 tmp_inode_name = kasprintf(GFP_KERNEL,
  "%s_%s_regs", accel_dev_name(adev), region->region_name);
 if (tmp_inode_name == NULL)
  return -ENOMEM;
 mur = kzalloc(sizeof(*mur), GFP_KERNEL);
 if (!mur) {
  fd = -ENOMEM;
  goto out_kfree_tmp_name;
 }
 mur->adev = adev;
 mur->size = region->size;
 if (is_virtual_dev) {
  mur->is_virtual_dev = true;
  mur->virt.base = vmalloc_base[region->base_BAR];
  mur->virt.pgoff = (region->BAR_offset) >> PAGE_SHIFT;
 } else {
  mur->is_virtual_dev = false;
  mur->phys.regs_paddr = BAR_phys_addr[region->base_BAR] +
     region->BAR_offset;
 }
 fd = anon_inode_getfd(tmp_inode_name, &accel_lib_user_regs_ctx_fd_fops,
    mur, O_RDWR);
 if (fd < 0)
  kfree(mur);
out_kfree_tmp_name:
 kfree(tmp_inode_name);
 return fd;
}
EXPORT_SYMBOL(accel_lib_map_user_regs);
