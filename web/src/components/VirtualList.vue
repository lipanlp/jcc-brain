<script setup lang="ts" generic="T">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
const props = withDefaults(defineProps<{ items:T[]; rowKey:(item:T)=>string|number; rowHeight?:number; overscan?:number; label?:string }>(), { rowHeight:24, overscan:10, label:'列表' });
const emit = defineEmits<{ scroll:[position:{top:number;distance:number}] }>();
defineSlots<{ default(props:{item:T;index:number}):unknown; empty?():unknown }>();
const viewport = ref<HTMLElement>();
const top = ref(0);
const height = ref(0);
const start = computed(() => Math.max(0,Math.min(Math.floor(top.value/props.rowHeight)-props.overscan, Math.max(0,props.items.length-1))));
const end = computed(() => Math.min(props.items.length,Math.ceil((top.value+height.value)/props.rowHeight)+props.overscan));
const visible = computed(() => props.items.slice(start.value,end.value));
function position() {
    const el=viewport.value;
    return {top:el?.scrollTop ?? 0,distance:el ? Math.max(0,el.scrollHeight-el.clientHeight-el.scrollTop) : 0};
}
function onScroll() { const value=position();top.value=value.top;emit('scroll',value); }
function scrollToBottom() { const el=viewport.value;if(el){el.scrollTop=el.scrollHeight;top.value=el.scrollTop;} }
const observer=new ResizeObserver(() => {height.value=viewport.value?.clientHeight ?? 0;onScroll();});
onMounted(() => {if(viewport.value){height.value=viewport.value.clientHeight;observer.observe(viewport.value);} });
onBeforeUnmount(() => observer.disconnect());
watch(() => props.items.length, async () => { await nextTick(); top.value=viewport.value?.scrollTop ?? 0; });
defineExpose({scrollToBottom,position});
</script>
<template>
  <div ref="viewport" class="virtual-list" tabindex="0" :aria-label="label" @scroll.passive="onScroll">
    <div v-if="items.length" class="virtual-space" :style="{height:`${items.length*rowHeight}px`}" role="list" :aria-label="label">
      <div class="virtual-window" :style="{transform:`translateY(${start*rowHeight}px)`}">
        <div v-for="(item,index) in visible" :key="rowKey(item)" role="listitem" :aria-posinset="start+index+1" :aria-setsize="items.length" :style="{height:`${rowHeight}px`}">
          <slot :item="item" :index="start+index" />
        </div>
      </div>
    </div>
    <slot v-else name="empty" />
  </div>
</template>
<style scoped>
.virtual-list {
  position: relative;
  flex: 1;
  min-height: 0;
  overflow: auto;
  overflow-anchor: none;
  scrollbar-gutter: stable;
}
.virtual-space {
  position: relative;
}
.virtual-window {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
}
</style>
