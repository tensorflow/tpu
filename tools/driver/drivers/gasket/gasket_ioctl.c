/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2020 Google LLC.
 */
#include "gasket_ioctl.h"
#include "gasket_constants.h"
#include "gasket_interrupt.h"
#include "gasket_logging.h"
#include "gasket_page_table.h"
#include <linux/fs.h>
#include <linux/google/gasket.h>
#include <linux/uaccess.h>
#define CREATE_TRACE_POINTS 
#include <trace/events/gasket_ioctl.h>
static uint gasket_ioctl_check_permissions(struct file *filp, uint cmd);
static int gasket_set_event_fd(struct gasket_dev *dev, ulong arg);
static int gasket_read_page_table_size(
 struct gasket_dev *gasket_dev, ulong arg);
static int gasket_read_simple_page_table_size(
 struct gasket_dev *gasket_dev, ulong arg);
static int gasket_partition_page_table(
 struct gasket_dev *gasket_dev, ulong arg);
static int gasket_map_buffers(struct gasket_dev *gasket_dev, ulong arg);
static int gasket_unmap_buffers(struct gasket_dev *gasket_dev, ulong arg);
static int gasket_register_interrupt(struct gasket_dev *gasket_dev, ulong arg);
static int gasket_unregister_interrupt(struct gasket_dev *gasket_dev,
 int interrupt);
long gasket_handle_ioctl(struct file *filp, uint cmd, ulong arg)
{
 struct gasket_dev *gasket_dev;
 int retval;
 gasket_dev = (struct gasket_dev *)filp->private_data;
 trace_gasket_ioctl_entry(accel_dev_name(&gasket_dev->accel_dev), cmd);
 if (gasket_get_ioctl_permissions_cb(gasket_dev)) {
  retval = gasket_get_ioctl_permissions_cb(gasket_dev)(
   filp, cmd, arg);
  if (retval < 0) {
   trace_gasket_ioctl_exit(-EPERM);
   return retval;
  } else if (retval == 0) {
   trace_gasket_ioctl_exit(-EPERM);
   return -EPERM;
  }
 } else if (!gasket_ioctl_check_permissions(filp, cmd)) {
  trace_gasket_ioctl_exit(-EPERM);
  return -EPERM;
 }
 switch (cmd) {
 case GASKET_IOCTL_RESET:
  trace_gasket_ioctl_integer_data(arg);
  retval = gasket_reset(gasket_dev, arg);
  break;
 case GASKET_IOCTL_SET_EVENTFD:
  retval = gasket_set_event_fd(gasket_dev, arg);
  break;
 case GASKET_IOCTL_CLEAR_EVENTFD:
  trace_gasket_ioctl_integer_data(arg);
  retval = legacy_gasket_interrupt_clear_eventfd(
   gasket_dev->interrupt_data, (int)arg);
  break;
 case GASKET_IOCTL_PARTITION_PAGE_TABLE:
  trace_gasket_ioctl_integer_data(arg);
  retval = gasket_partition_page_table(gasket_dev, arg);
  break;
 case GASKET_IOCTL_NUMBER_PAGE_TABLES:
  trace_gasket_ioctl_integer_data(gasket_dev->num_page_tables);
  if (copy_to_user((void __user *)arg,
       &gasket_dev->num_page_tables, sizeof(uint64_t)))
   retval = -EFAULT;
  else
   retval = 0;
  break;
 case GASKET_IOCTL_PAGE_TABLE_SIZE:
  retval = gasket_read_page_table_size(gasket_dev, arg);
  break;
 case GASKET_IOCTL_SIMPLE_PAGE_TABLE_SIZE:
  retval = gasket_read_simple_page_table_size(gasket_dev, arg);
  break;
 case GASKET_IOCTL_MAP_BUFFER:
  retval = gasket_map_buffers(gasket_dev, arg);
  break;
 case GASKET_IOCTL_UNMAP_BUFFER:
  retval = gasket_unmap_buffers(gasket_dev, arg);
  break;
 case GASKET_IOCTL_CLEAR_INTERRUPT_COUNTS:
  trace_gasket_ioctl_integer_data(0);
  retval = gasket_interrupt_reset_counts(gasket_dev);
  break;
 case GASKET_IOCTL_REGISTER_INTERRUPT:
  retval = gasket_register_interrupt(gasket_dev, arg);
  break;
 case GASKET_IOCTL_UNREGISTER_INTERRUPT:
  retval = gasket_unregister_interrupt(gasket_dev, (int)arg);
  break;
 default:
  trace_gasket_ioctl_integer_data(arg);
  gasket_log_warn(gasket_dev,
   "Unknown ioctl cmd=0x%x not caught by gasket_is_supported_ioctl",
   cmd);
  retval = -EINVAL;
  break;
 }
 trace_gasket_ioctl_exit(retval);
 return retval;
}
long gasket_is_supported_ioctl(uint cmd)
{
 switch (cmd) {
 case GASKET_IOCTL_RESET:
 case GASKET_IOCTL_SET_EVENTFD:
 case GASKET_IOCTL_CLEAR_EVENTFD:
 case GASKET_IOCTL_PARTITION_PAGE_TABLE:
 case GASKET_IOCTL_NUMBER_PAGE_TABLES:
 case GASKET_IOCTL_PAGE_TABLE_SIZE:
 case GASKET_IOCTL_SIMPLE_PAGE_TABLE_SIZE:
 case GASKET_IOCTL_MAP_BUFFER:
 case GASKET_IOCTL_UNMAP_BUFFER:
 case GASKET_IOCTL_CLEAR_INTERRUPT_COUNTS:
 case GASKET_IOCTL_REGISTER_INTERRUPT:
 case GASKET_IOCTL_UNREGISTER_INTERRUPT:
  return 1;
 default:
  return 0;
 }
}
static uint gasket_ioctl_check_permissions(struct file *filp, uint cmd)
{
 uint alive, root, device_owner;
 fmode_t read, write;
 struct gasket_dev *gasket_dev = (struct gasket_dev *)filp->private_data;
 alive = (gasket_dev->status == GASKET_STATUS_ALIVE);
 root = capable(CAP_SYS_ADMIN);
 read = filp->f_mode & FMODE_READ;
 write = filp->f_mode & FMODE_WRITE;
 device_owner = (gasket_dev->ownership.is_owned &&
   current->tgid == gasket_dev->ownership.owner);
 switch (cmd) {
 case GASKET_IOCTL_RESET:
 case GASKET_IOCTL_CLEAR_INTERRUPT_COUNTS:
  return root || (write && device_owner);
 case GASKET_IOCTL_PAGE_TABLE_SIZE:
 case GASKET_IOCTL_SIMPLE_PAGE_TABLE_SIZE:
 case GASKET_IOCTL_NUMBER_PAGE_TABLES:
  return root || read;
 case GASKET_IOCTL_PARTITION_PAGE_TABLE:
  return alive && (root || (write && device_owner));
 case GASKET_IOCTL_MAP_BUFFER:
 case GASKET_IOCTL_UNMAP_BUFFER:
  return alive && (root || (write && device_owner));
 case GASKET_IOCTL_CLEAR_EVENTFD:
 case GASKET_IOCTL_SET_EVENTFD:
 case GASKET_IOCTL_REGISTER_INTERRUPT:
 case GASKET_IOCTL_UNREGISTER_INTERRUPT:
  return alive && (root || (write && device_owner));
 }
 return 0;
}
static int gasket_register_interrupt(struct gasket_dev *gasket_dev, ulong arg)
{
 struct gasket_interrupt_mapping mapping;
 int ret;
 if (copy_from_user(&mapping, (void __user *)arg,
      sizeof(struct gasket_interrupt_mapping))) {
  return -EFAULT;
 }
 trace_gasket_ioctl_register_interrupt_data(mapping.interrupt,
  mapping.event_fd, mapping.bar_index, mapping.reg_offset);
 mutex_lock(&gasket_dev->mutex);
 ret = gasket_interrupt_register_mapping(
  gasket_dev, mapping.interrupt, mapping.event_fd,
  mapping.bar_index, mapping.reg_offset);
 mutex_unlock(&gasket_dev->mutex);
 return ret;
}
static int gasket_unregister_interrupt(struct gasket_dev *gasket_dev,
     int interrupt)
{
 int ret;
 trace_gasket_ioctl_integer_data(interrupt);
 mutex_lock(&gasket_dev->mutex);
 ret = gasket_interrupt_unregister_mapping(gasket_dev, interrupt);
 mutex_unlock(&gasket_dev->mutex);
 return ret;
}
static int gasket_set_event_fd(struct gasket_dev *gasket_dev, ulong arg)
{
 struct gasket_interrupt_eventfd die;
 if (copy_from_user(&die, (void __user *)arg,
      sizeof(struct gasket_interrupt_eventfd))) {
  return -EFAULT;
 }
 trace_gasket_ioctl_eventfd_data(die.interrupt, die.event_fd);
 return legacy_gasket_interrupt_set_eventfd(
  gasket_dev->interrupt_data, die.interrupt, die.event_fd);
}
static int gasket_read_page_table_size(struct gasket_dev *gasket_dev, ulong arg)
{
 int ret = 0;
 struct gasket_page_table_ioctl ibuf;
 if (copy_from_user(&ibuf, (void __user *)arg,
      sizeof(struct gasket_page_table_ioctl)))
  return -EFAULT;
 if (ibuf.page_table_index >= gasket_dev->num_page_tables)
  return -EFAULT;
 ibuf.size = gasket_page_table_num_entries(
  gasket_dev->page_table[ibuf.page_table_index]);
 trace_gasket_ioctl_page_table_data(ibuf.page_table_index, ibuf.size,
  ibuf.host_address, ibuf.device_address);
 if (copy_to_user((void __user *)arg, &ibuf, sizeof(ibuf)))
  return -EFAULT;
 return ret;
}
static int gasket_read_simple_page_table_size(
 struct gasket_dev *gasket_dev, ulong arg)
{
 int ret = 0;
 struct gasket_page_table_ioctl ibuf;
 if (copy_from_user(&ibuf, (void __user *)arg,
      sizeof(struct gasket_page_table_ioctl)))
  return -EFAULT;
 if (ibuf.page_table_index >= gasket_dev->num_page_tables)
  return -EFAULT;
 ibuf.size = gasket_page_table_num_simple_entries(
  gasket_dev->page_table[ibuf.page_table_index]);
 trace_gasket_ioctl_page_table_data(ibuf.page_table_index, ibuf.size,
  ibuf.host_address, ibuf.device_address);
 if (copy_to_user((void __user *)arg, &ibuf, sizeof(ibuf)))
  return -EFAULT;
 return ret;
}
static int gasket_partition_page_table(struct gasket_dev *gasket_dev, ulong arg)
{
 int ret;
 struct gasket_page_table_ioctl ibuf;
 uint max_page_table_size;
 if (copy_from_user(&ibuf, (void __user *)arg,
      sizeof(struct gasket_page_table_ioctl)))
  return -EFAULT;
 trace_gasket_ioctl_page_table_data(ibuf.page_table_index, ibuf.size,
  ibuf.host_address, ibuf.device_address);
 if (ibuf.page_table_index >= gasket_dev->num_page_tables)
  return -EFAULT;
 max_page_table_size = gasket_page_table_max_size(
  gasket_dev->page_table[ibuf.page_table_index]);
 if (ibuf.size > max_page_table_size) {
  gasket_log_error(gasket_dev,
   "Partition request 0x%llx too large, max is 0x%x.",
   ibuf.size, max_page_table_size);
  return -EINVAL;
 }
 mutex_lock(&gasket_dev->mutex);
 ret = gasket_page_table_partition(
  gasket_dev->page_table[ibuf.page_table_index], ibuf.size);
 mutex_unlock(&gasket_dev->mutex);
 return ret;
}
static int gasket_map_buffers(struct gasket_dev *gasket_dev, ulong arg)
{
 struct gasket_page_table_ioctl ibuf;
 if (copy_from_user(&ibuf, (void __user *)arg,
      sizeof(struct gasket_page_table_ioctl)))
  return -EFAULT;
 trace_gasket_ioctl_page_table_data(ibuf.page_table_index, ibuf.size,
  ibuf.host_address, ibuf.device_address);
 if (ibuf.page_table_index >= gasket_dev->num_page_tables)
  return -EFAULT;
 return gasket_page_table_map(
  gasket_dev->page_table[ibuf.page_table_index],
  ibuf.host_address, ibuf.device_address, ibuf.size);
}
static int gasket_unmap_buffers(struct gasket_dev *gasket_dev, ulong arg)
{
 struct gasket_page_table_ioctl ibuf;
 if (copy_from_user(&ibuf, (void __user *)arg,
      sizeof(struct gasket_page_table_ioctl)))
  return -EFAULT;
 trace_gasket_ioctl_page_table_data(ibuf.page_table_index, ibuf.size,
  ibuf.host_address, ibuf.device_address);
 if (ibuf.page_table_index >= gasket_dev->num_page_tables)
  return -EFAULT;
 return gasket_page_table_unmap(
  gasket_dev->page_table[ibuf.page_table_index],
  ibuf.device_address, ibuf.size);
}
