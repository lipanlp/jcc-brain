<script setup lang="ts">
import { computed } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { Settings } from 'lucide-vue-next';
import { navigation } from '../navigation';
import { openSettings } from '../composables/settings';
import { vWindowDrag } from '../composables/windowDrag';
const route = useRoute();
const router = useRouter();
const groups = computed(() => ['对局', '功能控制', '资料'].map(title => ({ title, items: navigation.filter(item => item.group === title) })));
</script>
<template>
  <aside class="sidebar">
    <div v-window-drag="'main'" class="traffic-drag" aria-hidden="true" />
    <div class="sidebar-navigation">
      <section v-for="group in groups" :key="group.title">
        <div class="sidebar-heading">{{ group.title }}</div>
        <nav :aria-label="group.title">
          <button v-for="item in group.items" :key="item.path" type="button" :data-nav="item.path" :aria-current="route.path === item.path ? 'page' : undefined" @click="router.push(item.path)">
            <component :is="item.icon" :size="18" /><span>{{ item.title }}</span>
          </button>
        </nav>
      </section>
    </div>
    <div class="sidebar-bottom">
      <button type="button" class="settings-button" @click="openSettings()">
        <Settings :size="17" />
        <span>通用设置</span>
      </button>
      <div class="app-signature">
        金铲铲大脑
        <span>对局辅助工作台</span>
      </div>
    </div>
  </aside>
</template>
<style scoped>
.sidebar {
  width: var(--sidebar-width);
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--separator);
  background: var(--sidebar);
  backdrop-filter: var(--material-blur);
}
.traffic-drag {
  height: var(--toolbar-height);
  flex-shrink: 0;
}
.sidebar-heading {
  color: var(--secondary);
  font-size: var(--font-xs);
  font-weight: 600;
  padding: var(--space-5) var(--space-6) var(--space-2);
}
nav {
  position: relative;
  margin: 0 var(--space-3);
  isolation: isolate;
}
.nav-capsule {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: var(--nav-height);
  border-radius: var(--radius-md);
  background: var(--accent);
  z-index: -1;
  transition: transform var(--duration-normal) var(--ease), opacity var(--duration-fast) var(--ease);
}
nav button {
  position: relative;
  display: flex;
  align-items: center;
  gap: var(--space-3);
  width: 100%;
  height: var(--nav-height);
  padding: 0 var(--space-3);
  border-radius: var(--radius-md);
  background: var(--transparent);
  font-size: var(--font-md);
  text-align: left;
}
nav button::before {
  position: absolute;
  content: '';
  inset: 0;
  background: var(--hover);
  border-radius: inherit;
  opacity: 0;
  transition: opacity var(--duration-fast) var(--ease);
  z-index: -1;
}
nav button:hover::before {
  opacity: 1;
}
nav button[aria-current] {
  color: var(--white);
  background: var(--accent);
  font-weight: 500;
}
kbd {
  font: inherit;
  font-size: var(--font-xs);
  margin-left: auto;
  opacity: .45;
}
.sidebar-navigation { overflow-y: auto; min-height: 0; }
.sidebar-bottom {
  margin-top: auto;
  padding: var(--space-3);
}
.settings-button {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-2) var(--space-3);
  background: var(--transparent);
  border-radius: var(--radius-md);
  width: 100%;
  text-align: left;
}
.settings-button:hover {
  background: var(--hover);
}
.app-signature {
  margin: var(--space-5) var(--space-3) var(--space-2);
  color: var(--secondary);
  font-size: var(--font-xs);
  font-weight: 500;
}
.app-signature span {
  display: block;
  color: var(--tertiary);
  font-weight: 400;
  margin-top: var(--space-half);
}
.settings-placeholder {
  padding: var(--space-8) 0;
  text-align: center;
  color: var(--secondary);
}
</style>
