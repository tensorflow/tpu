/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include <linux/dma-buf.h>
#include <linux/fs.h>
#include <linux/minmax.h>
#include <linux/scatterlist.h>
#include <linux/uaccess.h>
#include <linux/version.h>
#include "gasket_core.h"
#include "gasket_dmabuf.h"
#include "gasket_logging.h"
#define DMABUF_MMAP_MAXIMUM_BLOCK_SIZE (1 << 14)
struct gasket_dma_buf_object {
 struct gasket_dev *gasket_dev;
 struct gasket_dma_buf_device_data *device_data;
 u64 mmap_offset;
};
#if LINUX_VERSION_CODE <= KERNEL_VERSION(4, 18, 20)
static void *gasket_dma_buf_ops_map_atomic(
 struct dma_buf *dbuf, unsigned long page_num)
{
 return NULL;
}
static void *gasket_dma_buf_ops_page_map(
 struct dma_buf *dbuf, unsigned long page_num)
{
 return NULL;
}
#endif
static int gasket_dma_buf_do_remap_pfn(struct vm_area_struct *vma,
 u64 phys_addr, size_t size, size_t block_size)
{
 int ret;
 size_t mapped_size = 0;
 while (mapped_size < size) {
  block_size = min(block_size, (size - mapped_size));
  cond_resched();
  ret = io_remap_pfn_range(vma, vma->vm_start + mapped_size,
   (phys_addr + mapped_size) >> PAGE_SHIFT, block_size,
   vma->vm_page_prot);
  if (ret) {
   zap_vma_ptes(vma, vma->vm_start, mapped_size);
   return ret;
  }
  mapped_size += block_size;
 }
 return 0;
}
static int gasket_dma_buf_ops_mmap(
 struct dma_buf *dbuf, struct vm_area_struct *vma)
{
 int bar_index;
 u64 phys_addr;
 size_t vma_size;
 struct gasket_dev *gasket_dev;
 struct gasket_dma_buf_object *gasket_dbuf;
 int ret;
 gasket_dbuf = dbuf->priv;
 gasket_dev = gasket_dbuf->gasket_dev;
 if (vma->vm_start & (PAGE_SIZE - 1)) {
  gasket_log_error(gasket_dev,
   "Base address not page-aligned: 0x%p\n",
   (void *)vma->vm_start);
  return -EINVAL;
 }
 vma_size = vma->vm_end - vma->vm_start;
 if (vma_size & (PAGE_SIZE - 1)) {
  gasket_log_error(gasket_dev,
   "Mapping size not page-aligned: 0x%lx\n", vma_size);
  return -EINVAL;
 }
 bar_index =
  gasket_get_mmap_bar_index(gasket_dev, gasket_dbuf->mmap_offset);
 if (bar_index < 0) {
  gasket_log_error(gasket_dev, "failed to get mmap bar index: %d",
   bar_index);
  return -EINVAL;
 }
 vma->vm_flags &= ~(VM_MAYREAD | VM_MAYWRITE | VM_MAYEXEC | VM_MAYSHARE);
 vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
 phys_addr = gasket_dbuf->mmap_offset + (vma->vm_pgoff << PAGE_SHIFT) -
      gasket_dev->driver_desc->bar_descriptions[bar_index].base +
      gasket_dev->bar_data[bar_index].phys_base;
 ret = gasket_dma_buf_do_remap_pfn(
  vma, phys_addr, vma_size, DMABUF_MMAP_MAXIMUM_BLOCK_SIZE);
 if (ret) {
  gasket_log_error(
   gasket_dev, "Error remapping PFN range: %d", ret);
  return ret;
 }
 vma->vm_private_data = dbuf;
 vma->vm_pgoff = phys_addr >> PAGE_SHIFT;
 return 0;
}
static struct sg_table *gasket_dma_buf_ops_map(
 struct dma_buf_attachment *attachment,
 enum dma_data_direction direction)
{
 int bar_index;
 dma_addr_t addr;
 u64 phys_addr;
 struct dma_buf *dbuf;
 struct gasket_dev *gasket_dev;
 struct gasket_dma_buf_object *gasket_dbuf;
 struct sg_table *sgt;
 int ret;
 dbuf = attachment->dmabuf;
 gasket_dbuf = dbuf->priv;
 gasket_dev = gasket_dbuf->gasket_dev;
 bar_index =
  gasket_get_mmap_bar_index(gasket_dev, gasket_dbuf->mmap_offset);
 if (bar_index < 0) {
  gasket_log_error(gasket_dev, "failed to get mmap bar index: %d",
   bar_index);
  return ERR_PTR(bar_index);
 }
 sgt = kzalloc(sizeof(struct sg_table), GFP_KERNEL);
 if (!sgt) {
  gasket_log_error(gasket_dev,
   "failed to allocate sg_table for dma-buf map");
  return ERR_PTR(-ENOMEM);
 }
 ret = sg_alloc_table(sgt, 1, GFP_KERNEL);
 if (ret) {
  gasket_log_error(gasket_dev,
   "failed sg_alloc_table for dma-buf map: %d", ret);
  goto error_alloc_table;
 }
 phys_addr = gasket_dbuf->mmap_offset -
      gasket_dev->driver_desc->bar_descriptions[bar_index].base +
      gasket_dev->bar_data[bar_index].phys_base;
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 1, 0)
 addr = dma_map_resource(attachment->dev, phys_addr, dbuf->size,
  direction, DMA_ATTR_SKIP_CPU_SYNC);
 ret = dma_mapping_error(attachment->dev, addr);
 if (ret) {
  gasket_log_error(gasket_dev,
   "failed to dma_map the backing storage: %d", ret);
  sg_free_table(sgt);
  goto error_alloc_table;
 }
#else
 addr = phys_addr;
#endif
 sg_set_page(sgt->sgl, NULL, dbuf->size, 0);
 sg_dma_address(sgt->sgl) = addr;
 sg_dma_len(sgt->sgl) = dbuf->size;
 return sgt;
error_alloc_table:
 kfree(sgt);
 return ERR_PTR(ret);
}
static void gasket_dma_buf_ops_unmap(struct dma_buf_attachment *attachment,
 struct sg_table *sgt, enum dma_data_direction direction)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 1, 0)
 struct scatterlist *sg;
 int i;
 for_each_sg((sgt)->sgl, sg, (sgt)->orig_nents, i)
 {
  dma_unmap_resource(attachment->dev, sg_dma_address(sg),
   sg_dma_len(sg), direction, DMA_ATTR_SKIP_CPU_SYNC);
 }
#endif
 sg_free_table(sgt);
 kfree(sgt);
}
static void gasket_dma_buf_ops_release(struct dma_buf *dbuf)
{
 struct gasket_dev *gasket_dev;
 struct gasket_dma_buf_object *gasket_dbuf;
 gasket_dbuf = dbuf->priv;
 gasket_dev = gasket_dbuf->gasket_dev;
 if (gasket_dbuf->device_data && gasket_dbuf->device_data->release_cb) {
  gasket_dbuf->device_data->release_cb(gasket_dbuf->device_data);
 }
 kfree(gasket_dbuf);
}
static const struct dma_buf_ops gasket_dma_buf_ops = {
 .map_dma_buf = gasket_dma_buf_ops_map,
 .unmap_dma_buf = gasket_dma_buf_ops_unmap,
 .release = gasket_dma_buf_ops_release,
 .mmap = gasket_dma_buf_ops_mmap,
#if LINUX_VERSION_CODE <= KERNEL_VERSION(4, 18, 20)
 .map_atomic = gasket_dma_buf_ops_map_atomic,
 .map = gasket_dma_buf_ops_page_map,
#endif
};
struct dma_buf *gasket_create_mmap_dma_buf(struct gasket_dev *gasket_dev,
 u64 mmap_offset, size_t buf_size, int flags,
 struct gasket_dma_buf_device_data *device_data)
{
 struct gasket_dma_buf_object *gasket_dbuf;
 DEFINE_DMA_BUF_EXPORT_INFO(gasket_dma_buf_exp);
 gasket_dbuf = kzalloc(sizeof(*gasket_dbuf), GFP_KERNEL);
 if (!gasket_dbuf) {
  return ERR_PTR(-ENOMEM);
 }
 gasket_dbuf->mmap_offset = mmap_offset;
 gasket_dbuf->gasket_dev = gasket_dev;
 gasket_dbuf->device_data = device_data;
 gasket_dma_buf_exp.ops = &gasket_dma_buf_ops;
 gasket_dma_buf_exp.size = buf_size;
 gasket_dma_buf_exp.flags = flags;
 gasket_dma_buf_exp.priv = gasket_dbuf;
 return dma_buf_export(&gasket_dma_buf_exp);
}
EXPORT_SYMBOL(gasket_create_mmap_dma_buf);
