/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include "gasket_page_table.h"
#include <linux/dma-buf.h>
#include <linux/file.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/pagemap.h>
#include <linux/version.h>
#include <linux/vmalloc.h>
#include <linux/dma-resv.h>
#if LINUX_VERSION_CODE > KERNEL_VERSION(5, 16, 0)
MODULE_IMPORT_NS(DMA_BUF);
#endif
#include "gasket_logging.h"
#define GASKET_PAGES_PER_SUBTABLE 512
#define GASKET_SIMPLE_PAGE_SHIFT 12
#define GASKET_VALID_SLOT_FLAG 1
#define GASKET_EXTENDED_LVL0_SHIFT 21
#define GASKET_EXTENDED_LVL1_SHIFT 12
#define gasket_pg_tbl_error(pg_tbl,format,arg...) \
 gasket_log_error((pg_tbl)->gasket_dev, \
  "Page table ID %d: " format, (pg_tbl)->config->id, ##arg)
#define gasket_pg_tbl_warn(pg_tbl,format,arg...) \
 gasket_log_warn((pg_tbl)->gasket_dev, \
  "Page table ID %d: " format, (pg_tbl)->config->id, ##arg)
#define gasket_pg_tbl_debug(pg_tbl,format,arg...) \
 gasket_log_debug((pg_tbl)->gasket_dev, \
  "Page table ID %d: " format, (pg_tbl)->config->id, ##arg)
enum pte_status {
 PTE_FREE,
 PTE_INUSE,
 PTE_INUSE_PFN,
 PTE_INUSE_SGT,
};
struct gasket_sgt_mapping {
 struct sg_table *sgt;
 struct dma_buf_attachment *dbuf_attach;
 ulong dev_addr;
 size_t size;
 struct list_head entry;
};
struct gasket_page_table_entry {
 enum pte_status status;
 dma_addr_t dma_addr;
 struct page *page;
 uint pfn_num_pages;
 struct gasket_page_table_entry *sublevel;
};
struct gasket_page_table {
 struct gasket_dev *gasket_dev;
 const struct gasket_page_table_config *config;
 uint num_simple_entries;
 uint num_extended_entries;
 struct gasket_page_table_entry *entries;
 uint num_active_pages;
 u64 __iomem *base_slot;
 u64 __iomem *extended_offset_reg;
 u64 extended_flag;
 struct mutex mutex;
 struct list_head sgt_mappings;
};
static int gasket_map_simple_pages(
 struct gasket_page_table *pg_tbl, ulong host_addr,
 ulong dev_addr, uint num_pages);
static int gasket_map_extended_pages(
 struct gasket_page_table *pg_tbl, ulong host_addr,
 ulong dev_addr, uint num_pages);
static int gasket_map_sgt_simple_pages(
 struct gasket_page_table *pg_tbl, struct gasket_sgt_mapping *mapping);
static int gasket_map_sgt_extended_pages(
 struct gasket_page_table *pg_tbl, struct gasket_sgt_mapping *mapping);
static struct gasket_sgt_mapping *gasket_page_table_import_dma_buf(
 struct gasket_page_table *pg_tbl, int dma_buf_fd);
static void gasket_page_table_detach_sgt_mapping(
 struct gasket_page_table *pg_tbl, struct gasket_sgt_mapping *mapping);
static int gasket_page_table_validate_sgt_mappings(
 struct gasket_page_table *pg_tbl, ulong dev_addr, uint num_pages);
static int gasket_page_table_detach_sgt_mappings(
 struct gasket_page_table *pg_tbl, ulong dev_addr, uint num_pages);
static size_t gasket_sgt_get_contiguous_size(
 struct gasket_page_table *pg_tbl, struct sg_table *sgt);
static int gasket_perform_mapping(
 struct gasket_page_table *pg_tbl,
 struct gasket_page_table_entry *pte_base, u64 __iomem *att_base,
 ulong host_addr, ulong dev_addr, uint num_pages, int is_simple_mapping);
static int gasket_alloc_simple_entries(
 struct gasket_page_table *pg_tbl, ulong dev_addr, uint num_pages);
static int gasket_alloc_extended_entries(
 struct gasket_page_table *pg_tbl, ulong dev_addr, uint num_entries);
static int gasket_alloc_extended_subtable(struct gasket_page_table *pg_tbl,
 struct gasket_page_table_entry *pte, u64 __iomem *att_reg);
static int gasket_page_table_unmap_nolock(
 struct gasket_page_table *pg_tbl, ulong start_addr, uint num_pages);
static void gasket_page_table_unmap_all_nolock(
 struct gasket_page_table *pg_tbl);
static int gasket_unmap_simple_pages(
 struct gasket_page_table *pg_tbl, ulong start_addr, uint num_pages);
static int gasket_unmap_extended_pages(
 struct gasket_page_table *pg_tbl, ulong start_addr, uint num_pages,
 bool strict);
static int gasket_perform_unmapping(struct gasket_page_table *pg_tbl,
 struct gasket_page_table_entry *pte_base, u64 __iomem *att_base,
 ulong dev_addr, uint num_pages, int is_simple_mapping,
 int *num_unmapped);
static void gasket_free_extended_subtable(struct gasket_page_table *pg_tbl,
 struct gasket_page_table_entry *pte, u64 __iomem *att_reg);
static int gasket_release_page(struct page *page);
static inline int gasket_addr_is_simple(
 struct gasket_page_table *pg_tbl, ulong addr);
static int gasket_is_simple_dev_addr_bad(
 struct gasket_page_table *pg_tbl, ulong dev_addr, uint num_pages);
static int gasket_is_extended_dev_addr_bad(
 struct gasket_page_table *pg_tbl, ulong dev_addr, uint num_pages);
static int gasket_is_pte_range_free(
 struct gasket_page_table_entry *pte, uint num_entries);
static void gasket_page_table_garbage_collect_nolock(
 struct gasket_page_table *pg_tbl);
static ulong gasket_components_to_dev_address(struct gasket_page_table *pg_tbl,
 int is_simple, uint page_index, uint offset);
static int gasket_simple_page_idx(
 struct gasket_page_table *pg_tbl, ulong dev_addr);
static ulong gasket_extended_lvl0_page_idx(
 struct gasket_page_table *pg_tbl, ulong dev_addr);
static ulong gasket_extended_lvl1_page_idx(
 struct gasket_page_table *pg_tbl, ulong dev_addr);
#if LINUX_VERSION_CODE < KERNEL_VERSION(6, 1, 0)
struct vm_area_struct * _find_vma(struct vm_area_struct *vma) {
   return vma->vm_next;
}
#else
struct vm_area_struct * _find_vma(struct vm_area_struct *vma) {
  return find_vma(vma->vm_mm, vma->vm_end);
}
#endif
int gasket_page_table_init(struct gasket_page_table **ppg_tbl,
 const struct gasket_bar_data *bar_data,
 const struct gasket_page_table_config *page_table_config,
 struct gasket_dev *gasket_dev)
{
 ulong bytes;
 struct gasket_page_table *pg_tbl;
 ulong total_entries = page_table_config->total_entries;
 if (total_entries == ULONG_MAX) {
  gasket_log_error(gasket_dev,
   "Error reading page table size. Initializing page table with size 0.");
  total_entries = 0;
 }
 gasket_log_debug(gasket_dev,
  "Attempting to initialize page table of size 0x%lx.",
  total_entries);
 gasket_log_debug(gasket_dev,
  "Table has base reg 0x%llx, extended offset reg 0x%llx.",
  page_table_config->base_reg,
  page_table_config->extended_reg);
 *ppg_tbl = kzalloc(sizeof(**ppg_tbl), GFP_KERNEL);
 if (!*ppg_tbl) {
  gasket_log_error(gasket_dev, "No memory for page tables.");
  return -ENOMEM;
 }
 pg_tbl = *ppg_tbl;
 bytes = total_entries * sizeof(struct gasket_page_table_entry);
 if (bytes != 0) {
  pg_tbl->entries = vmalloc(bytes);
  if (!pg_tbl->entries) {
   gasket_log_error(gasket_dev,
    "No memory for address translation metadata.");
   kfree(pg_tbl);
   *ppg_tbl = NULL;
   return -ENOMEM;
  }
  memset(pg_tbl->entries, 0, bytes);
 }
 mutex_init(&pg_tbl->mutex);
 pg_tbl->config = page_table_config;
 if (pg_tbl->config->mode == GASKET_PAGE_TABLE_MODE_NORMAL ||
  pg_tbl->config->mode == GASKET_PAGE_TABLE_MODE_SIMPLE) {
  pg_tbl->num_simple_entries = total_entries;
  pg_tbl->num_extended_entries = 0;
  pg_tbl->extended_flag = 1ull << page_table_config->extended_bit;
 } else {
  pg_tbl->num_simple_entries = 0;
  pg_tbl->num_extended_entries = total_entries;
  pg_tbl->extended_flag = 0;
 }
 pg_tbl->num_active_pages = 0;
 pg_tbl->base_slot = (u64 __iomem *)&(
  bar_data->virt_base[page_table_config->base_reg]);
 pg_tbl->extended_offset_reg = (u64 __iomem *)&(
  bar_data->virt_base[page_table_config->extended_reg]);
 pg_tbl->gasket_dev = gasket_dev;
 INIT_LIST_HEAD(&pg_tbl->sgt_mappings);
 gasket_log_debug(gasket_dev, "Page table initialized successfully.");
 return 0;
}
void gasket_page_table_cleanup(struct gasket_page_table *pg_tbl)
{
 gasket_page_table_garbage_collect(pg_tbl);
 vfree(pg_tbl->entries);
 pg_tbl->entries = NULL;
 kfree(pg_tbl);
}
int gasket_page_table_partition(
 struct gasket_page_table *pg_tbl, uint num_simple_entries)
{
 int i, start;
 mutex_lock(&pg_tbl->mutex);
 if (pg_tbl->config->mode != GASKET_PAGE_TABLE_MODE_NORMAL) {
  gasket_pg_tbl_error(
   pg_tbl, "This page table is not configurable!");
  return -EINVAL;
 }
 if (num_simple_entries > pg_tbl->config->total_entries) {
  mutex_unlock(&pg_tbl->mutex);
  return -EINVAL;
 }
 gasket_page_table_garbage_collect_nolock(pg_tbl);
 start = min(pg_tbl->num_simple_entries, num_simple_entries);
 for (i = start; i < pg_tbl->config->total_entries; i++) {
  if (pg_tbl->entries[i].status != PTE_FREE) {
   gasket_pg_tbl_error(pg_tbl, "entry %d is not free", i);
   mutex_unlock(&pg_tbl->mutex);
   return -EBUSY;
  }
 }
 pg_tbl->num_simple_entries = num_simple_entries;
 pg_tbl->num_extended_entries =
  pg_tbl->config->total_entries - num_simple_entries;
 writeq(num_simple_entries, pg_tbl->extended_offset_reg);
 mutex_unlock(&pg_tbl->mutex);
 return 0;
}
EXPORT_SYMBOL(gasket_page_table_partition);
int gasket_page_table_dma_buf_map(
 struct gasket_page_table *pg_tbl, int dma_buf_fd, ulong dev_addr)
{
 int ret;
 uint num_pages;
 struct gasket_sgt_mapping *mapping;
 struct dma_buf *dbuf;
 mapping = gasket_page_table_import_dma_buf(pg_tbl, dma_buf_fd);
 if (IS_ERR(mapping))
  return PTR_ERR(mapping);
 mutex_lock(&pg_tbl->mutex);
 INIT_LIST_HEAD(&mapping->entry);
 mapping->dev_addr = dev_addr;
 dbuf = mapping->dbuf_attach->dmabuf;
 if (mapping->size !=
  gasket_sgt_get_contiguous_size(pg_tbl, mapping->sgt)) {
  gasket_pg_tbl_error(
   pg_tbl, "failed to map non-contiguous scatterlist");
  ret = -EINVAL;
  goto dma_buf_map_failed;
 }
 num_pages = mapping->size >> PAGE_SHIFT;
 if (gasket_page_table_is_dev_addr_bad(
      pg_tbl, dev_addr, mapping->size)) {
  ret = -EINVAL;
  goto dma_buf_map_failed;
 }
 if (gasket_addr_is_simple(pg_tbl, dev_addr))
  ret = gasket_map_sgt_simple_pages(pg_tbl, mapping);
 else
  ret = gasket_map_sgt_extended_pages(pg_tbl, mapping);
 if (ret) {
  gasket_pg_tbl_error(pg_tbl, "failed to map dma_buf: %d", ret);
  goto dma_buf_map_failed;
 }
 list_add_tail(&mapping->entry, &pg_tbl->sgt_mappings);
 mutex_unlock(&pg_tbl->mutex);
 return ret;
dma_buf_map_failed:
 gasket_page_table_detach_sgt_mapping(pg_tbl, mapping);
 mutex_unlock(&pg_tbl->mutex);
 return ret;
}
EXPORT_SYMBOL(gasket_page_table_dma_buf_map);
int gasket_page_table_map(struct gasket_page_table *pg_tbl, ulong host_addr,
 ulong dev_addr, ulong bytes)
{
 int ret = 0;
 uint num_pages;
 mutex_lock(&pg_tbl->mutex);
 if (gasket_page_table_are_addrs_bad(pg_tbl, host_addr, dev_addr,
         bytes)) {
  ret = -EINVAL;
 } else {
  num_pages = bytes / PAGE_SIZE;
  if (gasket_addr_is_simple(pg_tbl, dev_addr)) {
   ret = gasket_map_simple_pages(
    pg_tbl, host_addr, dev_addr, num_pages);
  } else {
   ret = gasket_map_extended_pages(
    pg_tbl, host_addr, dev_addr, num_pages);
  }
 }
 mutex_unlock(&pg_tbl->mutex);
 return ret;
}
EXPORT_SYMBOL(gasket_page_table_map);
int gasket_page_table_unmap(
 struct gasket_page_table *pg_tbl, ulong dev_addr, ulong bytes)
{
 int ret = 0;
 uint num_pages;
 mutex_lock(&pg_tbl->mutex);
 if (gasket_page_table_is_dev_addr_bad(pg_tbl, dev_addr, bytes)) {
  ret = -EINVAL;
 } else {
  num_pages = bytes / PAGE_SIZE;
  ret = gasket_page_table_unmap_nolock(pg_tbl, dev_addr,
           num_pages);
 }
 mutex_unlock(&pg_tbl->mutex);
 return ret;
}
EXPORT_SYMBOL(gasket_page_table_unmap);
static void gasket_page_table_unmap_all_nolock(struct gasket_page_table *pg_tbl)
{
 WARN_ON(gasket_unmap_simple_pages(pg_tbl,
  gasket_components_to_dev_address(pg_tbl, 1, 0, 0),
  pg_tbl->num_simple_entries));
 WARN_ON(gasket_unmap_extended_pages(pg_tbl,
  gasket_components_to_dev_address(pg_tbl, 0, 0, 0),
  pg_tbl->num_extended_entries * GASKET_PAGES_PER_SUBTABLE,
  false));
}
void gasket_page_table_unmap_all(struct gasket_page_table *pg_tbl)
{
 mutex_lock(&pg_tbl->mutex);
 gasket_page_table_unmap_all_nolock(pg_tbl);
 mutex_unlock(&pg_tbl->mutex);
}
EXPORT_SYMBOL(gasket_page_table_unmap_all);
void gasket_page_table_reset(struct gasket_page_table *pg_tbl)
{
 mutex_lock(&pg_tbl->mutex);
 gasket_page_table_unmap_all_nolock(pg_tbl);
 if (pg_tbl->config->mode == GASKET_PAGE_TABLE_MODE_NORMAL)
  writeq(pg_tbl->config->total_entries,
    pg_tbl->extended_offset_reg);
 mutex_unlock(&pg_tbl->mutex);
}
void gasket_page_table_garbage_collect(struct gasket_page_table *pg_tbl)
{
 mutex_lock(&pg_tbl->mutex);
 gasket_page_table_garbage_collect_nolock(pg_tbl);
 mutex_unlock(&pg_tbl->mutex);
}
EXPORT_SYMBOL(gasket_page_table_garbage_collect);
int gasket_page_table_are_addrs_bad(struct gasket_page_table *pg_tbl,
 ulong host_addr, ulong dev_addr, ulong bytes)
{
 struct vm_area_struct *vma;
 ulong current_ptr = host_addr;
 ulong end_ptr = host_addr + bytes;
 ulong required_perms = VM_READ | VM_WRITE;
 if (host_addr & (PAGE_SIZE - 1)) {
  gasket_pg_tbl_error(pg_tbl,
   "host mapping address 0x%lx must be page aligned",
   host_addr);
  return 1;
 }
 down_read(&current->mm->mmap_lock);
 vma = find_vma(current->mm, current_ptr);
 if (unlikely(!vma)) {
  gasket_pg_tbl_error(pg_tbl,
   "No VMA found that satisfies 0x%lx < vm_end",
   current_ptr);
  goto fail;
 }
 while (current_ptr < end_ptr) {
  if (vma->vm_start > current_ptr || vma->vm_end <= current_ptr) {
   gasket_pg_tbl_error(pg_tbl,
    "Found VMA [0x%lx, 0x%lx) does not include the host VM address: 0x%lx",
    vma->vm_start, vma->vm_end, current_ptr);
   goto fail;
  }
  if ((vma->vm_flags & required_perms) != required_perms) {
   gasket_pg_tbl_error(pg_tbl,
    "Process must have read/write perms (0x%lx) to map address 0x%lx into device (VMA: [0x%lx, 0x%lx), flags: 0x%lx)",
    required_perms, current_ptr, vma->vm_flags,
    vma->vm_start, vma->vm_end);
   goto fail;
  }
  if ((vma->vm_flags & VM_DONTCOPY) != VM_DONTCOPY) {
   gasket_pg_tbl_warn(pg_tbl,
    "Mapping VMA [0x%lx, 0x%lx) that does not have VM_DONTCOPY set. This may lead to memory corruption after fork.",
    vma->vm_start, vma->vm_end);
   gasket_mapped_unforkable_page(pg_tbl->gasket_dev);
  }
  if (end_ptr <= vma->vm_end)
   break;
  current_ptr = vma->vm_end;
  vma = _find_vma(vma);
  if (!vma) {
   gasket_pg_tbl_error(pg_tbl,
    "No more VMAs after [0x%lx, 0x%lx) to search through",
    vma->vm_start, vma->vm_end);
   goto fail;
  }
 }
 up_read(&current->mm->mmap_lock);
 return gasket_page_table_is_dev_addr_bad(pg_tbl, dev_addr, bytes);
fail:
 up_read(&current->mm->mmap_lock);
 gasket_pg_tbl_error(pg_tbl,
  "Failure while checking host range [0x%lx, 0x%lx)", host_addr,
  host_addr + bytes);
 return -EFAULT;
}
EXPORT_SYMBOL(gasket_page_table_are_addrs_bad);
int gasket_page_table_is_dev_addr_bad(
 struct gasket_page_table *pg_tbl, ulong dev_addr, ulong bytes)
{
 uint num_pages = bytes / PAGE_SIZE;
 if (bytes & (PAGE_SIZE - 1)) {
  gasket_pg_tbl_error(pg_tbl,
   "mapping size 0x%lX must be page aligned", bytes);
  return 1;
 }
 if (num_pages == 0) {
  gasket_pg_tbl_error(pg_tbl,
   "requested mapping is less than one page: %lu / %lu",
   bytes, PAGE_SIZE);
  return 1;
 }
 if (gasket_addr_is_simple(pg_tbl, dev_addr))
  return gasket_is_simple_dev_addr_bad(
   pg_tbl, dev_addr, num_pages);
 else
  return gasket_is_extended_dev_addr_bad(
   pg_tbl, dev_addr, num_pages);
}
EXPORT_SYMBOL(gasket_page_table_is_dev_addr_bad);
uint gasket_page_table_max_size(struct gasket_page_table *pg_tbl)
{
 if (unlikely(!pg_tbl)) {
  gasket_nodev_error("Passed a null page table.");
  return 0;
 }
 return pg_tbl->config->total_entries;
}
EXPORT_SYMBOL(gasket_page_table_max_size);
uint gasket_page_table_num_entries(struct gasket_page_table *pg_tbl)
{
 if (unlikely(!pg_tbl)) {
  gasket_nodev_error("Passed a null page table.");
  return 0;
 }
 return pg_tbl->num_simple_entries + pg_tbl->num_extended_entries;
}
EXPORT_SYMBOL(gasket_page_table_num_entries);
uint gasket_page_table_num_simple_entries(struct gasket_page_table *pg_tbl)
{
 if (unlikely(!pg_tbl)) {
  gasket_nodev_error("Passed a null page table.");
  return 0;
 }
 return pg_tbl->num_simple_entries;
}
EXPORT_SYMBOL(gasket_page_table_num_simple_entries);
uint gasket_page_table_num_extended_entries(struct gasket_page_table *pg_tbl)
{
 if (unlikely(!pg_tbl)) {
  gasket_nodev_error("Passed a null page table.");
  return 0;
 }
 return pg_tbl->num_extended_entries;
}
EXPORT_SYMBOL(gasket_page_table_num_extended_entries);
uint gasket_page_table_num_active_pages(struct gasket_page_table *pg_tbl)
{
 if (unlikely(!pg_tbl)) {
  gasket_nodev_error("Passed a null page table.");
  return 0;
 }
 return pg_tbl->num_active_pages;
}
EXPORT_SYMBOL(gasket_page_table_num_active_pages);
int gasket_page_table_system_status(struct gasket_page_table *pg_tbl)
{
 if (unlikely(!pg_tbl)) {
  gasket_nodev_error("Passed a null page table.");
  return GASKET_STATUS_LAMED;
 }
 if (gasket_page_table_num_entries(pg_tbl) == 0) {
  gasket_log_error(pg_tbl->gasket_dev,
   "Page table size is 0.");
  return GASKET_STATUS_LAMED;
 }
 return GASKET_STATUS_ALIVE;
}
static int gasket_map_simple_pages(struct gasket_page_table *pg_tbl,
 ulong host_addr, ulong dev_addr, uint num_pages)
{
 int ret;
 uint slot_idx = gasket_simple_page_idx(pg_tbl, dev_addr);
 ret = gasket_alloc_simple_entries(pg_tbl, dev_addr, num_pages);
 if (ret) {
  gasket_pg_tbl_error(pg_tbl,
   "page table slots %u (@ 0x%lx) to %u are not available",
   slot_idx, dev_addr, slot_idx + num_pages - 1);
  return ret;
 }
 ret = gasket_perform_mapping(pg_tbl, pg_tbl->entries + slot_idx,
  pg_tbl->base_slot + slot_idx, host_addr, dev_addr, num_pages,
  1);
 if (ret)
  gasket_page_table_unmap_nolock(pg_tbl, dev_addr, num_pages);
 return ret;
}
static int gasket_map_extended_pages(
 struct gasket_page_table *pg_tbl, ulong host_addr,
 ulong dev_addr, uint num_pages)
{
 int ret;
 ulong dev_addr_start = dev_addr;
 ulong dev_addr_end;
 uint slot_idx, remain, len;
 struct gasket_page_table_entry *pte;
 u64 __iomem *slot_base;
 ret = gasket_alloc_extended_entries(pg_tbl, dev_addr, num_pages);
 if (ret) {
  dev_addr_end = dev_addr + (num_pages / PAGE_SIZE) - 1;
  gasket_pg_tbl_error(pg_tbl,
   "page table slots (%lu,%lu) (@ 0x%lx) to (%lu,%lu) are not available",
   gasket_extended_lvl0_page_idx(pg_tbl, dev_addr),
   gasket_extended_lvl1_page_idx(pg_tbl, dev_addr),
   dev_addr,
   gasket_extended_lvl0_page_idx(pg_tbl, dev_addr_end),
   gasket_extended_lvl1_page_idx(pg_tbl, dev_addr_end));
  return ret;
 }
 remain = num_pages;
 slot_idx = gasket_extended_lvl1_page_idx(pg_tbl, dev_addr);
 pte = pg_tbl->entries + pg_tbl->num_simple_entries +
       gasket_extended_lvl0_page_idx(pg_tbl, dev_addr);
 while (remain > 0) {
  len = min(remain, GASKET_PAGES_PER_SUBTABLE - slot_idx);
  slot_base =
   (u64 __iomem *)(page_address(pte->page));
  ret = gasket_perform_mapping(pg_tbl, pte->sublevel + slot_idx,
   slot_base + slot_idx, host_addr, dev_addr, len, 0);
  if (ret) {
   gasket_page_table_unmap_nolock(
    pg_tbl, dev_addr_start, num_pages - remain);
   return ret;
  }
  remain -= len;
  slot_idx = 0;
  pte++;
  host_addr += len * PAGE_SIZE;
  dev_addr += len * PAGE_SIZE;
 }
 return 0;
}
static bool use_hpa_for_pfn_mappings(struct pci_dev *dev, ulong host_addr,
    ulong size)
{
 return false;
}
static int perform_pfn_mapping(struct gasket_page_table *pg_tbl,
 struct gasket_page_table_entry *ptes, u64 __iomem *slots,
 ulong host_addr, ulong dev_addr, uint num_pages, int is_simple_mapping)
{
 int ret = 0;
 int i;
 struct vm_area_struct *vma;
 const size_t size = (size_t)num_pages << PAGE_SHIFT;
 unsigned long vma_offset;
        bool always_use_hpa = false;
 phys_addr_t host_phys_addr;
 dma_addr_t host_dma_addr = 0;
 dma_addr_t page_dma_addr;
 dma_addr_t device_dma_addr;
 down_read(&current->mm->mmap_lock);
 vma = find_vma(current->mm, host_addr);
 if (vma == NULL) {
  ret = -EFAULT;
  goto error_mm;
 }
 if (!(vma->vm_flags & VM_PFNMAP)) {
  gasket_pg_tbl_error(pg_tbl,
   "VMA region is not VM_PFNMAP: host_addr=%#lx, size=%#lx, vm_flags=%#lx, vm_pgoff=%#lx",
   host_addr, size, vma->vm_flags, vma->vm_pgoff);
  ret = -EINVAL;
  goto error_mm;
 }
 if (host_addr + size > vma->vm_end) {
  gasket_pg_tbl_error(pg_tbl,
   "VMA region does not cover the requested range: host_addr=%#lx, size=%#lx, vm_start=%#lx, vm_end=%#lx, vm_pgoff=%#lx",
   host_addr, size, vma->vm_start, vma->vm_end,
   vma->vm_pgoff);
  ret = -EINVAL;
  goto error_mm;
 }
 vma_offset = host_addr - vma->vm_start;
 host_phys_addr = (vma->vm_pgoff << PAGE_SHIFT) + vma_offset;
 up_read(&current->mm->mmap_lock);
 gasket_pg_tbl_debug(pg_tbl,
  "%s: virt_addr_valid=%d, host_phys_addr=%#llx, iommu_ops=%p\n",
  __func__, virt_addr_valid(host_phys_addr), host_phys_addr,
  pg_tbl->gasket_dev->pci_dev->dev.bus->iommu_ops);
        always_use_hpa = use_hpa_for_pfn_mappings(pg_tbl->gasket_dev->pci_dev,
                                                   host_phys_addr, size);
 if (gasket_pci_is_iommu_enabled(pg_tbl->gasket_dev->pci_dev) &&
            !always_use_hpa) {
  ret = dma_mapping_error(
   &pg_tbl->gasket_dev->pci_dev->dev, host_dma_addr);
  if (ret) {
   gasket_pg_tbl_error(pg_tbl,
    "Error allocating IOVA for PFN region for DMA: %d",
    ret);
   return ret;
  }
  ret = dma_mapping_error(
   &pg_tbl->gasket_dev->pci_dev->dev, page_dma_addr);
  if (ret) {
   gasket_pg_tbl_error(pg_tbl,
    "Error mapping IOVA to PFN region for DMA: %d",
    ret);
   goto error;
  }
  WARN_ON(host_dma_addr != page_dma_addr);
 } else {
#ifdef CONFIG_VIRT_TO_BUS
  ret = 0;
  host_dma_addr = virt_to_bus(phys_to_virt(host_phys_addr));
#else
  gasket_pg_tbl_error(pg_tbl,
   "Mapping PFN regions for DMA is not supported");
  return -EINVAL;
#endif
 }
 if (pg_tbl->config->map_buffer_cb) {
  ret = pg_tbl->config->map_buffer_cb(
   pg_tbl->gasket_dev, pg_tbl->config->id,
   host_dma_addr, dev_addr, num_pages);
  if (ret) {
   gasket_pg_tbl_error(pg_tbl,
    "Map buffer callback error for dev_addr=%#lx, host_addr=%#lx, host_dma_addr=%#llx, size=%#lx: %d",
    dev_addr, host_addr, host_dma_addr, size, ret);
   goto error;
  }
 }
 gasket_pg_tbl_debug(pg_tbl,
  "Mapped VMA region: dev_addr=%#lx, host_addr=%#lx, size=%#lx, vm_start=%#lx, vm_end=%#lx, vm_pgoff=%#lx, host_ptr=%#llx, dma_addr=%#llx",
  dev_addr, host_addr, size, vma->vm_start, vma->vm_end,
  vma->vm_pgoff, host_phys_addr, host_dma_addr);
 ptes[0].pfn_num_pages = num_pages;
 page_dma_addr = host_dma_addr;
 for (i = 0; i < num_pages; ++i, page_dma_addr += PAGE_SIZE) {
  ptes[i].dma_addr = page_dma_addr;
  device_dma_addr = page_dma_addr | GASKET_VALID_SLOT_FLAG;
  if (is_simple_mapping)
   writeq(device_dma_addr, &slots[i]);
  else
   ((u64 __force *)slots)[i] = device_dma_addr;
  ptes[i].status = PTE_INUSE_PFN;
 }
 pg_tbl->num_active_pages += num_pages;
 return 0;
error:
 if (gasket_pci_is_iommu_enabled(pg_tbl->gasket_dev->pci_dev) &&
     host_dma_addr && !always_use_hpa) {}
 return ret;
error_mm:
 up_read(&current->mm->mmap_lock);
 return ret;
}
static int gasket_perform_mapping(struct gasket_page_table *pg_tbl,
 struct gasket_page_table_entry *ptes, u64 __iomem *slots,
 ulong host_addr, ulong dev_addr, uint num_pages, int is_simple_mapping)
{
 struct device *dev = &pg_tbl->gasket_dev->pci_dev->dev;
 int i, ret, num_dma_mapped = 0;
 uint pages_to_free = 0;
 dma_addr_t dma_addr;
 ulong page_addr;
 struct page **pages;
 if (num_pages == 0)
  return 0;
 if (host_addr & (PAGE_SIZE - 1)) {
  gasket_pg_tbl_error(pg_tbl,
   "host address 0x%lx is not page-aligned (this should never get here)!",
   host_addr);
  return -EINVAL;
 }
 pages = kmalloc_array(num_pages, sizeof(struct page *), GFP_KERNEL);
 if (!pages)
  return -ENOMEM;
 ret = get_user_pages_fast(host_addr, num_pages, 1, pages);
 if (ret == -EFAULT) {
  ret = perform_pfn_mapping(pg_tbl, ptes, slots, host_addr,
       dev_addr, num_pages,
       is_simple_mapping);
  goto exit;
 }
 if (ret != num_pages) {
  gasket_pg_tbl_error(pg_tbl,
   "Failed to map buffer: host_addr=%#lx, ret=%d, num_pages=%d",
   host_addr, ret, num_pages);
  if (ret >= 0) {
   pages_to_free = ret;
   ret = -ENOMEM;
  }
  goto fail;
 }
 pg_tbl->num_active_pages += num_pages;
 pages_to_free = num_pages;
 for (i = 0; i < num_pages; i++) {
  page_addr = host_addr + i * PAGE_SIZE;
  ptes[i].page = pages[i];
  ptes[i].dma_addr = dma_map_page(dev,
   pages[i], 0, PAGE_SIZE, DMA_BIDIRECTIONAL);
  ret = dma_mapping_error(dev, ptes[i].dma_addr);
  if (ret) {
   gasket_pg_tbl_error(pg_tbl,
    "Error mapping page for DMA: %d", ret);
   ret = -EFAULT;
   goto fail;
  }
  num_dma_mapped++;
  mb();
  if (pg_tbl->config->map_buffer_cb) {
   ret = pg_tbl->config->map_buffer_cb(
    pg_tbl->gasket_dev, pg_tbl->config->id,
    ptes[i].dma_addr, dev_addr, 1);
   if (ret) {
    gasket_pg_tbl_error(pg_tbl,
     "Map buffer callback error for dev_addr=%#lx: %d",
     dev_addr, ret);
    goto fail;
   }
   dev_addr += PAGE_SIZE;
  }
  dma_addr = (ptes[i].dma_addr) | GASKET_VALID_SLOT_FLAG;
  if (is_simple_mapping)
   writeq(dma_addr, &slots[i]);
  else
   ((u64 __force *)slots)[i] = dma_addr;
  ptes[i].status = PTE_INUSE;
 }
 ret = 0;
 goto exit;
fail:
 for (i = 0; i < pages_to_free; i++) {
  if (i < num_dma_mapped) {
   dma_unmap_page(dev, ptes[i].dma_addr, PAGE_SIZE,
    DMA_BIDIRECTIONAL);
  }
  gasket_release_page(pages[i]);
  ptes[i].status = PTE_FREE;
 }
exit:
 kfree(pages);
 return ret;
}
static int gasket_alloc_simple_entries(
 struct gasket_page_table *pg_tbl, ulong dev_addr, uint num_pages)
{
 if (!gasket_is_pte_range_free(
      pg_tbl->entries + gasket_simple_page_idx(pg_tbl, dev_addr),
      num_pages))
  return -EBUSY;
 return 0;
}
static int gasket_alloc_extended_entries(
 struct gasket_page_table *pg_tbl, ulong dev_addr, uint num_entries)
{
 int ret = 0;
 uint remain, subtable_slot_idx, len;
 struct gasket_page_table_entry *pte;
 u64 __iomem *slot;
 remain = num_entries;
 subtable_slot_idx = gasket_extended_lvl1_page_idx(pg_tbl, dev_addr);
 pte = pg_tbl->entries + pg_tbl->num_simple_entries +
  gasket_extended_lvl0_page_idx(pg_tbl, dev_addr);
 slot = pg_tbl->base_slot + pg_tbl->num_simple_entries +
  gasket_extended_lvl0_page_idx(pg_tbl, dev_addr);
 while (remain > 0) {
  len = min(remain,
   GASKET_PAGES_PER_SUBTABLE - subtable_slot_idx);
  if (pte->status == PTE_FREE) {
   ret = gasket_alloc_extended_subtable(pg_tbl, pte, slot);
   if (ret) {
    gasket_pg_tbl_error(pg_tbl,
     "no memory for extended addr subtable");
    return ret;
   }
  } else {
   if (!gasket_is_pte_range_free(
        pte->sublevel + subtable_slot_idx, len))
    return -EBUSY;
  }
  remain -= len;
  subtable_slot_idx = 0;
  pte++;
  slot++;
 }
 return 0;
}
static int gasket_alloc_extended_subtable(struct gasket_page_table *pg_tbl,
 struct gasket_page_table_entry *pte, u64 __iomem *slot)
{
 struct device *dev = &pg_tbl->gasket_dev->pci_dev->dev;
 ulong page_addr, subtable_bytes;
 dma_addr_t dma_addr;
 page_addr = get_zeroed_page(GFP_KERNEL);
 if (!page_addr)
  return -ENOMEM;
 pte->page = virt_to_page((void *)page_addr);
 subtable_bytes = sizeof(struct gasket_page_table_entry) *
    GASKET_PAGES_PER_SUBTABLE;
 pte->sublevel = vmalloc(subtable_bytes);
 if (!pte->sublevel)
  goto fail_subtable_allocate;
 memset(pte->sublevel, 0, subtable_bytes);
 pte->dma_addr = dma_map_page(dev, pte->page, 0, PAGE_SIZE,
  DMA_BIDIRECTIONAL);
 if (unlikely(dma_mapping_error(dev, pte->dma_addr)))
  goto fail_dma_map;
 mb();
 dma_addr = pte->dma_addr | GASKET_VALID_SLOT_FLAG;
 writeq(dma_addr, slot);
 pte->status = PTE_INUSE;
 return 0;
fail_dma_map:
 vfree(pte->sublevel);
fail_subtable_allocate:
 free_page(page_addr);
 memset(pte, 0, sizeof(struct gasket_page_table_entry));
 return -ENOMEM;
}
static int gasket_page_table_unmap_nolock(
 struct gasket_page_table *pg_tbl, ulong dev_addr, uint num_pages)
{
 if (!num_pages)
  return 0;
 if (gasket_addr_is_simple(pg_tbl, dev_addr))
  return gasket_unmap_simple_pages(pg_tbl, dev_addr, num_pages);
 else
  return gasket_unmap_extended_pages(
   pg_tbl, dev_addr, num_pages, true);
}
static int gasket_unmap_simple_pages(
 struct gasket_page_table *pg_tbl, ulong dev_addr, uint num_pages)
{
 int ret;
 uint slot = gasket_simple_page_idx(pg_tbl, dev_addr);
 ret = gasket_page_table_validate_sgt_mappings(
  pg_tbl, dev_addr, num_pages);
 if (ret) {
  gasket_pg_tbl_error(pg_tbl,
   "Partial unmapping of scatterlist is not allowed: %d",
   ret);
  return ret;
 }
 ret = gasket_perform_unmapping(pg_tbl, pg_tbl->entries + slot,
  pg_tbl->base_slot + slot, dev_addr, num_pages, 1, NULL);
 if (ret)
  return ret;
 gasket_page_table_detach_sgt_mappings(pg_tbl, dev_addr, num_pages);
 return ret;
}
static void perform_sgt_dma_addr_mapping(struct gasket_page_table *pg_tbl,
 struct gasket_page_table_entry *ptes, u64 __iomem *slots,
 dma_addr_t dma_addr, uint num_pages, int is_simple_mapping)
{
 int i;
 dma_addr_t page_dma_addr;
 dma_addr_t device_dma_addr;
 gasket_pg_tbl_debug(pg_tbl, "Mapping dma_addr=%#llx, iommu_ops=%p\n",
  dma_addr, pg_tbl->gasket_dev->pci_dev->dev.bus->iommu_ops);
 page_dma_addr = dma_addr;
 for (i = 0; i < num_pages; ++i, page_dma_addr += PAGE_SIZE) {
  ptes[i].dma_addr = page_dma_addr;
  device_dma_addr = page_dma_addr | GASKET_VALID_SLOT_FLAG;
  if (is_simple_mapping)
   writeq(device_dma_addr, &slots[i]);
  else
   ((u64 __force *)slots)[i] = device_dma_addr;
  ptes[i].status = PTE_INUSE_SGT;
 }
 pg_tbl->num_active_pages += num_pages;
}
static size_t gasket_sgt_get_contiguous_size(
 struct gasket_page_table *pg_tbl, struct sg_table *sgt)
{
 int i;
 size_t sz = 0;
 dma_addr_t start;
 struct scatterlist *sg;
 start = sg_dma_address(sgt->sgl);
 for_each_sg(sgt->sgl, sg, sgt->nents, i) {
  if (!IS_ALIGNED(sg_dma_address(sg), PAGE_SIZE) ||
   !IS_ALIGNED(sg_dma_len(sg), PAGE_SIZE)) {
   gasket_pg_tbl_error(pg_tbl,
    "Entry in sg_table is not page aligned: dma_address: 0x%llx size: 0x%x",
    sg_dma_address(sg), sg->length);
   break;
  }
  if (sg_dma_address(sg) != start) {
   break;
  }
  start += sg_dma_len(sg);
  sz += sg_dma_len(sg);
 }
 return sz;
}
static void gasket_page_table_dma_buf_move_notify(
 struct dma_buf_attachment *attachment)
{
 struct gasket_sgt_mapping *mapping;
 struct gasket_page_table *pg_tbl;
 pg_tbl = attachment->importer_priv;
 if (!pg_tbl) {
  gasket_pg_tbl_warn(pg_tbl,
   "dma_buf move notify received with invalid pg_tbl.");
  return;
 }
 mutex_lock(&pg_tbl->mutex);
 list_for_each_entry(mapping, &pg_tbl->sgt_mappings, entry) {
  if (mapping->dbuf_attach == attachment) {
   gasket_pg_tbl_warn(pg_tbl,
    "dma_buf attachment moved: %p", attachment);
   break;
  }
 }
 mutex_unlock(&pg_tbl->mutex);
}
static const struct dma_buf_attach_ops gasket_dma_buf_attach_ops = {
 .allow_peer2peer = true,
 .move_notify = gasket_page_table_dma_buf_move_notify,
};
static struct gasket_sgt_mapping *gasket_page_table_import_dma_buf(
 struct gasket_page_table *pg_tbl, int dma_buf_fd)
{
 int ret;
 struct gasket_dev *gasket_dev;
 struct gasket_sgt_mapping *mapping;
 struct dma_buf *dbuf;
 gasket_dev = pg_tbl->gasket_dev;
 dbuf = dma_buf_get(dma_buf_fd);
 if (IS_ERR(dbuf)) {
  ret = PTR_ERR(dbuf);
  gasket_log_error(gasket_dev,
   "failed to get dma_buf using filedescriptor: %d", ret);
  return ERR_PTR(ret);
 }
 if (dbuf->size == 0) {
  gasket_log_error(
   gasket_dev, "Cannot import dma_buf with zero size");
  ret = -EINVAL;
  goto failed_mapping_alloc;
 }
 ret = 0;
 mapping = kzalloc(sizeof(*mapping), GFP_KERNEL);
 if (!mapping) {
  gasket_log_error(gasket_dev,
   "failed to allocate memory for mapping struct");
  ret = -ENOMEM;
  goto failed_mapping_alloc;
 }
 mapping->dbuf_attach = dma_buf_dynamic_attach(dbuf,
  &gasket_dev->pci_dev->dev, &gasket_dma_buf_attach_ops, pg_tbl);
 if (IS_ERR(mapping->dbuf_attach)) {
  ret = PTR_ERR(mapping->dbuf_attach);
  gasket_log_error(
   gasket_dev, "failed to attach to the dma_buf: %d", ret);
  goto failed_attach;
 }
 mapping->size = dbuf->size;
 dma_resv_lock(dbuf->resv, NULL);
 mapping->sgt =
  dma_buf_map_attachment(mapping->dbuf_attach, DMA_BIDIRECTIONAL);
 dma_resv_unlock(dbuf->resv);
 if (IS_ERR(mapping->sgt)) {
  ret = PTR_ERR(mapping->sgt);
  gasket_log_error(gasket_dev,
   "failed to map dma_buf attachment: %d", ret);
  goto failed_map;
 }
 return mapping;
failed_map:
 dma_buf_detach(dbuf, mapping->dbuf_attach);
failed_attach:
 kfree(mapping);
failed_mapping_alloc:
 dma_buf_put(dbuf);
 return ERR_PTR(ret);
}
static void gasket_page_table_detach_sgt_mapping(
 struct gasket_page_table *pg_tbl, struct gasket_sgt_mapping *mapping)
{
 struct dma_buf *dbuf;
 lockdep_assert_held(&pg_tbl->mutex);
 if (!list_empty(&mapping->entry))
  list_del_init(&mapping->entry);
 if (mapping->dbuf_attach) {
  dbuf = mapping->dbuf_attach->dmabuf;
  dma_resv_lock(dbuf->resv, NULL);
  dma_buf_unmap_attachment(
   mapping->dbuf_attach, mapping->sgt, DMA_BIDIRECTIONAL);
  dma_resv_unlock(dbuf->resv);
  dma_buf_detach(dbuf, mapping->dbuf_attach);
  dma_buf_put(dbuf);
 }
 kfree(mapping);
}
static int gasket_map_sgt_simple_pages(
 struct gasket_page_table *pg_tbl, struct gasket_sgt_mapping *mapping)
{
 int ret;
 ulong dev_addr;
 dma_addr_t dma_address;
 uint num_pages, slot_idx;
 lockdep_assert_held(&pg_tbl->mutex);
 dev_addr = mapping->dev_addr;
 slot_idx = gasket_simple_page_idx(pg_tbl, dev_addr);
 dma_address = sg_dma_address(mapping->sgt->sgl);
 num_pages = mapping->size >> PAGE_SHIFT;
 ret = gasket_alloc_simple_entries(pg_tbl, dev_addr, num_pages);
 if (ret) {
  gasket_pg_tbl_error(pg_tbl,
   "page table slots %u (@ 0x%lx) to %u are not available",
   slot_idx, dev_addr, slot_idx + num_pages - 1);
  return ret;
 }
 perform_sgt_dma_addr_mapping(pg_tbl, pg_tbl->entries + slot_idx,
  pg_tbl->base_slot + slot_idx, dma_address, num_pages, 1);
 if (pg_tbl->config->map_buffer_cb) {
  ret = pg_tbl->config->map_buffer_cb(pg_tbl->gasket_dev,
   pg_tbl->config->id, dma_address, dev_addr, num_pages);
  if (ret) {
   gasket_pg_tbl_error(pg_tbl,
    "Map buffer callback error for dev_addr=%#lx, dma_addr=%#llx, num_pages=%d: %d",
    dev_addr, dma_address, num_pages, ret);
   gasket_page_table_unmap_nolock(
    pg_tbl, dev_addr, num_pages);
  }
 }
 return ret;
}
static int gasket_map_sgt_extended_pages(
 struct gasket_page_table *pg_tbl, struct gasket_sgt_mapping *mapping)
{
 int ret;
 ulong dev_addr = mapping->dev_addr;
 ulong dev_addr_end;
 uint remain_pages, num_pages, slot_idx, len;
 struct gasket_page_table_entry *pte;
 dma_addr_t dma_address;
 u64 __iomem *slot_base;
 lockdep_assert_held(&pg_tbl->mutex);
 dma_address = sg_dma_address(mapping->sgt->sgl);
 num_pages = mapping->size >> PAGE_SHIFT;
 ret = gasket_alloc_extended_entries(pg_tbl, dev_addr, num_pages);
 if (ret) {
  dev_addr_end = dev_addr + (num_pages * PAGE_SIZE) - 1;
  gasket_pg_tbl_error(pg_tbl,
   "page table slots (%lu,%lu) (@ 0x%lx) to (%lu,%lu) are not available",
   gasket_extended_lvl0_page_idx(pg_tbl, dev_addr),
   gasket_extended_lvl1_page_idx(pg_tbl, dev_addr),
   dev_addr,
   gasket_extended_lvl0_page_idx(pg_tbl, dev_addr_end),
   gasket_extended_lvl1_page_idx(pg_tbl, dev_addr_end));
  return ret;
 }
 slot_idx = gasket_extended_lvl1_page_idx(pg_tbl, dev_addr);
 pte = pg_tbl->entries + pg_tbl->num_simple_entries +
       gasket_extended_lvl0_page_idx(pg_tbl, dev_addr);
 remain_pages = num_pages;
 while (remain_pages > 0) {
  len = min(remain_pages, GASKET_PAGES_PER_SUBTABLE - slot_idx);
  slot_base = (u64 __iomem *)(page_address(pte->page));
  perform_sgt_dma_addr_mapping(pg_tbl, pte->sublevel + slot_idx,
   slot_base + slot_idx, dma_address, len, 0);
  slot_idx = 0;
  pte++;
  remain_pages -= len;
  dma_address += len * PAGE_SIZE;
  dev_addr += len * PAGE_SIZE;
 }
 if (pg_tbl->config->map_buffer_cb) {
  dev_addr = mapping->dev_addr;
  dma_address = sg_dma_address(mapping->sgt->sgl);
  ret = pg_tbl->config->map_buffer_cb(pg_tbl->gasket_dev,
   pg_tbl->config->id, dma_address, dev_addr, num_pages);
  if (ret) {
   gasket_pg_tbl_error(pg_tbl,
    "Map buffer callback error for dev_addr=%#lx, dma_addr=%#llx, num_pages=%d: %d",
    dev_addr, dma_address, num_pages, ret);
   gasket_page_table_unmap_nolock(
    pg_tbl, dev_addr, num_pages);
  }
 }
 return ret;
}
static int gasket_unmap_extended_pages(struct gasket_page_table *pg_tbl,
 ulong dev_addr, uint num_pages, bool strict)
{
 uint slot_idx, remain, len;
 ulong start_dev_addr = dev_addr;
 struct gasket_page_table_entry *pte;
 u64 __iomem *slot_base;
 int ret = 0;
 int num_unmapped = 0;
 ret = gasket_page_table_validate_sgt_mappings(
  pg_tbl, dev_addr, num_pages);
 if (ret) {
  gasket_pg_tbl_error(pg_tbl,
   "Partial unmapping of scatterlist is not allowed: %d",
   ret);
  return ret;
 }
 remain = num_pages;
 slot_idx = gasket_extended_lvl1_page_idx(pg_tbl, dev_addr);
 pte = pg_tbl->entries + pg_tbl->num_simple_entries +
       gasket_extended_lvl0_page_idx(pg_tbl, dev_addr);
 while (remain > 0) {
  len = min(remain, GASKET_PAGES_PER_SUBTABLE - slot_idx);
  WARN_ON(pte->status == PTE_INUSE_PFN);
  if (pte->status == PTE_INUSE || pte->status == PTE_INUSE_SGT) {
   slot_base = (u64 __iomem *)(page_address(pte->page));
   ret = gasket_perform_unmapping(
    pg_tbl, pte->sublevel + slot_idx,
    slot_base + slot_idx, dev_addr, len, 0,
    &num_unmapped);
  }
  remain -= len;
  slot_idx = 0;
  pte++;
  dev_addr += len * PAGE_SIZE;
 }
 if (strict && num_unmapped != num_pages) {
  gasket_pg_tbl_error(pg_tbl,
   "Strict checking: incorrect number of pages freed; expected %d, got %d",
   num_pages, num_unmapped);
  return -ENOENT;
 }
 gasket_page_table_detach_sgt_mappings(
  pg_tbl, start_dev_addr, num_pages);
 return ret;
}
static int gasket_page_table_validate_sgt_mappings(
 struct gasket_page_table *pg_tbl, ulong dev_addr, uint num_pages)
{
 ulong mapping_dev_addr_end;
 ulong dev_addr_end = dev_addr + ((ulong)num_pages << PAGE_SHIFT) - 1;
 struct gasket_sgt_mapping *mapping;
 list_for_each_entry(mapping, &pg_tbl->sgt_mappings, entry) {
  mapping_dev_addr_end = mapping->dev_addr + mapping->size - 1;
  if (mapping->dev_addr > dev_addr_end ||
   dev_addr > mapping_dev_addr_end) {
   continue;
  }
  if (mapping->dev_addr < dev_addr ||
   mapping_dev_addr_end > dev_addr_end) {
   return -EINVAL;
  }
 }
 return 0;
}
static int gasket_page_table_detach_sgt_mappings(
 struct gasket_page_table *pg_tbl, ulong dev_addr, uint num_pages)
{
 int ret, i;
 uint sgt_num_pages, sgl_num_pages;
 ulong mapping_dev_addr_end;
 ulong dev_addr_end = dev_addr + ((ulong)num_pages << PAGE_SHIFT);
 struct gasket_sgt_mapping *mapping, *mapping2;
 struct scatterlist *sgl;
 ulong current_dev_addr;
 list_for_each_entry_safe(
  mapping, mapping2, &pg_tbl->sgt_mappings, entry)
 {
  sgt_num_pages = mapping->size >> PAGE_SHIFT;
  mapping_dev_addr_end = mapping->dev_addr + mapping->size;
  if (mapping->dev_addr >= dev_addr_end ||
   dev_addr >= mapping_dev_addr_end) {
   continue;
  }
  if (pg_tbl->config->unmap_buffer_cb) {
   current_dev_addr = dev_addr;
   for_each_sg(mapping->sgt->sgl, sgl, mapping->sgt->nents,
    i) {
    sgl_num_pages = sg_dma_len(sgl) >> PAGE_SHIFT;
    ret = pg_tbl->config->unmap_buffer_cb(
     pg_tbl->gasket_dev, pg_tbl->config->id,
     current_dev_addr, sgl_num_pages);
    if (ret) {
     gasket_pg_tbl_error(pg_tbl,
      "Unmap buffer callback error for (dev_addr=%#lx, num_pages=%u): %d (ignoring)",
      current_dev_addr, sgl_num_pages,
      ret);
    }
    current_dev_addr += sgl_num_pages * PAGE_SIZE;
   }
  }
  gasket_page_table_detach_sgt_mapping(pg_tbl, mapping);
  pg_tbl->num_active_pages -= sgt_num_pages;
 }
 return 0;
}
static int validate_pfn_mappings(struct gasket_page_table_entry *ptes,
    uint num_pages)
{
 int i, j;
 for (i = 0; i < num_pages; ++i) {
  if (ptes[i].status != PTE_INUSE_PFN)
   continue;
  if (ptes[i].pfn_num_pages == 0)
   return -EINVAL;
  if (ptes[i].pfn_num_pages > num_pages - i)
   return -EINVAL;
  WARN_ON(!ptes[i].dma_addr);
  for (j = 1; j < ptes[i].pfn_num_pages; ++j) {
   WARN_ON(ptes[i + j].status != PTE_INUSE_PFN);
   WARN_ON(ptes[i + j].pfn_num_pages);
  }
  i += ptes[i].pfn_num_pages;
 }
 return 0;
}
static int gasket_perform_unmapping(struct gasket_page_table *pg_tbl,
 struct gasket_page_table_entry *ptes, u64 __iomem *slots,
 ulong dev_addr, uint num_pages, int is_simple_mapping,
 int *num_unmapped)
{
 struct device *dev = &pg_tbl->gasket_dev->pci_dev->dev;
 int i;
 int ret;
        bool used_hpa;
 ret = validate_pfn_mappings(ptes, num_pages);
 if (ret)
  return ret;
 for (i = 0; i < num_pages; i++) {
  if (is_simple_mapping &&
      readq((void *) &slots[i]) != 0) {
   writeq(0, &slots[i]);
   if (num_unmapped)
    (*num_unmapped)++;
  } else if (!is_simple_mapping && slots[i] != 0) {
   ((u64 __force *) slots)[i] = 0;
   if (num_unmapped)
    (*num_unmapped)++;
  }
  mb();
  if (ptes[i].status == PTE_INUSE) {
   if (pg_tbl->config->unmap_buffer_cb) {
    ret = pg_tbl->config->unmap_buffer_cb(
     pg_tbl->gasket_dev, pg_tbl->config->id,
     dev_addr, 1);
    if (ret) {
     gasket_pg_tbl_error(pg_tbl,
      "Unmap buffer callback error for dev_addr=%#lx: %d (ignoring)",
      dev_addr, ret);
    }
    dev_addr += PAGE_SIZE;
   }
   if (ptes[i].dma_addr)
    dma_unmap_page(dev, ptes[i].dma_addr, PAGE_SIZE,
     DMA_BIDIRECTIONAL);
   if (gasket_release_page(ptes[i].page))
    --pg_tbl->num_active_pages;
  } else if (ptes[i].status == PTE_INUSE_PFN &&
      ptes[i].pfn_num_pages) {
   if (pg_tbl->config->unmap_buffer_cb) {
    ret = pg_tbl->config->unmap_buffer_cb(
     pg_tbl->gasket_dev, pg_tbl->config->id,
     dev_addr, ptes[i].pfn_num_pages);
    if (ret) {
     gasket_pg_tbl_error(pg_tbl,
      "Unmap buffer callback error for dev_addr=%#lx: %d (ignoring)",
      dev_addr, ret);
    }
   }
#ifdef CONFIG_VIRT_TO_BUS
                        used_hpa = use_hpa_for_pfn_mappings(
                            pg_tbl->gasket_dev->pci_dev,
                            virt_to_phys(bus_to_virt(ptes[i].dma_addr)),
                            ptes[i].pfn_num_pages << PAGE_SHIFT);
#else
                        used_hpa = false;
#endif
                        if (gasket_pci_is_iommu_enabled(
     pg_tbl->gasket_dev->pci_dev) &&
                            !used_hpa) {
   }
   pg_tbl->num_active_pages -= ptes[i].pfn_num_pages;
  }
  ptes[i].status = PTE_FREE;
  memset(&ptes[i], 0, sizeof(struct gasket_page_table_entry));
 }
 return 0;
}
static void gasket_free_extended_subtable(struct gasket_page_table *pg_tbl,
 struct gasket_page_table_entry *pte, u64 __iomem *slot)
{
 struct device *dev = &pg_tbl->gasket_dev->pci_dev->dev;
 pte->status = PTE_FREE;
 writeq(0, slot);
 mb();
 if (pte->dma_addr)
  dma_unmap_page(dev, pte->dma_addr, PAGE_SIZE,
   DMA_BIDIRECTIONAL);
 vfree(pte->sublevel);
 if (pte->page)
  free_page((ulong)page_address(pte->page));
 memset(pte, 0, sizeof(struct gasket_page_table_entry));
}
static int gasket_release_page(struct page *page)
{
 if (!page)
  return 0;
 if (!PageReserved(page))
  SetPageDirty(page);
 put_page(page);
 return 1;
}
static inline int gasket_addr_is_simple(
 struct gasket_page_table *pg_tbl, ulong addr)
{
 if (pg_tbl->config->mode == GASKET_PAGE_TABLE_MODE_SIMPLE)
  return true;
 else if (pg_tbl->config->mode == GASKET_PAGE_TABLE_MODE_EXTENDED)
  return false;
 else
  return !((addr) & (pg_tbl)->extended_flag);
}
static int gasket_is_simple_dev_addr_bad(
 struct gasket_page_table *pg_tbl, ulong dev_addr, uint num_pages)
{
 ulong page_offset = dev_addr & (PAGE_SIZE - 1);
 ulong page_index =
  (dev_addr / PAGE_SIZE) & (pg_tbl->config->total_entries - 1);
 if (gasket_components_to_dev_address(
      pg_tbl, 1, page_index, page_offset) != dev_addr) {
  gasket_pg_tbl_error(
   pg_tbl, "address is invalid, 0x%lX", dev_addr);
  return 1;
 }
 if (page_index >= pg_tbl->num_simple_entries) {
  gasket_pg_tbl_error(pg_tbl,
   "starting slot at %lu is too large, max is < %u",
   page_index, pg_tbl->num_simple_entries);
  return 1;
 }
 if (page_index + num_pages > pg_tbl->num_simple_entries) {
  gasket_pg_tbl_error(pg_tbl,
   "ending slot at %lu is too large, max is <= %u",
   page_index + num_pages, pg_tbl->num_simple_entries);
  return 1;
 }
 return 0;
}
static int gasket_is_extended_dev_addr_bad(
 struct gasket_page_table *pg_tbl, ulong dev_addr, uint num_pages)
{
 ulong subpage_start, subpage_end;
 ulong page_offset;
 ulong page_global_idx;
 ulong mask = ~(1ull << pg_tbl->config->extended_bit);
 page_offset = dev_addr & (PAGE_SIZE - 1);
 page_global_idx = (dev_addr / PAGE_SIZE) &
  (pg_tbl->config->total_entries * GASKET_PAGES_PER_SUBTABLE - 1);
 if (gasket_components_to_dev_address(
      pg_tbl, 0, page_global_idx, page_offset) != dev_addr) {
  gasket_pg_tbl_error(
   pg_tbl, "address is invalid, 0x%p", (void *)dev_addr);
  return 1;
 }
 subpage_start = (dev_addr & mask) / PAGE_SIZE;
 subpage_end = subpage_start + num_pages;
 if (subpage_end <= subpage_start ||
     subpage_end > pg_tbl->num_extended_entries *
     GASKET_PAGES_PER_SUBTABLE) {
  gasket_pg_tbl_error(pg_tbl,
    "Invalid device address range, (0x%lx - 0x%lx)",
    dev_addr, dev_addr + num_pages * PAGE_SIZE - 1);
  return 1;
 }
 return 0;
}
static int gasket_is_pte_range_free(
 struct gasket_page_table_entry *ptes, uint num_entries)
{
 int i;
 for (i = 0; i < num_entries; i++) {
  if (ptes[i].status != PTE_FREE)
   return 0;
 }
 return 1;
}
static void gasket_page_table_garbage_collect_nolock(
 struct gasket_page_table *pg_tbl)
{
 struct gasket_page_table_entry *pte;
 u64 __iomem *slot;
 for (pte = pg_tbl->entries + pg_tbl->num_simple_entries,
  slot = pg_tbl->base_slot + pg_tbl->num_simple_entries;
  pte < pg_tbl->entries + pg_tbl->config->total_entries;
  pte++, slot++) {
  if (pte->status == PTE_INUSE) {
   if (gasket_is_pte_range_free(
        pte->sublevel, GASKET_PAGES_PER_SUBTABLE))
    gasket_free_extended_subtable(
     pg_tbl, pte, slot);
  }
 }
}
static ulong gasket_components_to_dev_address(struct gasket_page_table *pg_tbl,
 int is_simple, uint page_index, uint offset)
{
 ulong lvl0_index, lvl1_index;
 if (is_simple) {
  lvl0_index = page_index & (pg_tbl->config->total_entries - 1);
  return (lvl0_index << GASKET_SIMPLE_PAGE_SHIFT) | offset;
 }
 lvl0_index = page_index / GASKET_PAGES_PER_SUBTABLE;
 lvl1_index = page_index & (GASKET_PAGES_PER_SUBTABLE - 1);
 return (pg_tbl)->extended_flag |
        (lvl0_index << GASKET_EXTENDED_LVL0_SHIFT) |
        (lvl1_index << GASKET_EXTENDED_LVL1_SHIFT) | offset;
}
static int gasket_simple_page_idx(
 struct gasket_page_table *pg_tbl, ulong dev_addr)
{
 return (dev_addr >> GASKET_SIMPLE_PAGE_SHIFT) &
        (pg_tbl->config->total_entries - 1);
}
static ulong gasket_extended_lvl0_page_idx(
 struct gasket_page_table *pg_tbl, ulong dev_addr)
{
 return (dev_addr >> GASKET_EXTENDED_LVL0_SHIFT) &
        (pg_tbl->config->total_entries - 1);
}
static ulong gasket_extended_lvl1_page_idx(
 struct gasket_page_table *pg_tbl, ulong dev_addr)
{
 return (dev_addr >> GASKET_EXTENDED_LVL1_SHIFT) &
        (GASKET_PAGES_PER_SUBTABLE - 1);
}
