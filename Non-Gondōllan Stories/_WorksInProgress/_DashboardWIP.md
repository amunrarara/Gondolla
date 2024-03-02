---
cssclasses:
  - dashboard
---
---
cssclasses:
  - dashboard
banner: "![[PortalOfTheSunGod_lowres.png]]
"
# Vault Info
- 🗄️ Recent file updates
 `$=dv.list(dv.pages('').sort(f=>f.file.mtime.ts,"desc").limit(4).file.link)`
- 〽️ Stats
	-  File Count: `$=dv.pages().length`