/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2020 Google LLC.
 */
#ifndef _UAPI_LINUX_ACCEL_IOCTL_H
#define _UAPI_LINUX_ACCEL_IOCTL_H 
#include <linux/ioctl.h>
#include <linux/types.h>
#define ACCEL_IOCTL 0xE4
#define ACCEL_VERSION_MAJOR 1
#define ACCEL_VERSION_MINOR 0
#define ACCEL_VERSION_PATCH 0
struct accel_ioctl_version {
 __u32 major;
 __u32 minor;
 __u32 patch;
};
#define ACCEL_IOCTL_VERSION \
 _IOR(ACCEL_IOCTL, 0, struct accel_ioctl_version)
struct accel_ioctl_open_queue {
 __u32 flags;
 __u32 queue_type;
 __u32 entry_size;
 __u32 nr_entries;
 __s64 fd;
};
#define ACCEL_IOCTL_OPEN_QUEUE \
 _IOWR(ACCEL_IOCTL, 1, struct accel_ioctl_open_queue)
#define ACCEL_QT_TO_DEVICE (0U << 31)
#define ACCEL_QT_FROM_DEVICE (1U << 31)
struct accel_ioctl_open_context {
 __s64 queue_fd[2];
 __u64 target_id;
 __u64 context_offset;
 __u64 context_size;
 __s64 context_fd;
};
#define ACCEL_IOCTL_OPEN_CONTEXT \
 _IOWR(ACCEL_IOCTL, 2, struct accel_ioctl_open_context)
struct accel_ioctl_map_buffer {
 __u64 buffer_size;
 __u64 host_vaddr;
 __u64 device_vaddr;
};
#define ACCEL_IOCTL_MAP_BUFFER \
 _IOWR(ACCEL_IOCTL, 4, struct accel_ioctl_map_buffer)
struct accel_ioctl_unmap_buffer {
 __u64 buffer_size;
 __u64 device_vaddr;
};
#define ACCEL_IOCTL_UNMAP_BUFFER \
 _IOW(ACCEL_IOCTL, 5, struct accel_ioctl_unmap_buffer)
struct accel_ioctl_map_user_regs {
 __u64 region_id;
 __u64 region_size;
};
#define ACCEL_IOCTL_MAP_USER_REGS \
 _IOW(ACCEL_IOCTL, 6, struct accel_ioctl_map_user_regs)
#endif
