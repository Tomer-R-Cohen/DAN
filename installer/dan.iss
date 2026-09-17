; DAN Windows installer (Inno Setup 6). Built by scripts\build_installer.ps1, which passes
; /DSourceDir=<staged package> /DOutputDir=<folder> /DAppVersion=<x.y.z> /DSetupName=<file>.
; Per-user install: no administrator rights needed.

#ifndef AppVersion
  #define AppVersion "1.1.0"
#endif
#ifndef SetupName
  #define SetupName "DAN-Setup"
#endif

[Setup]
AppId={{6B0E5B8C-3F0B-4C7E-9E2A-DA0DA0DA0D01}
AppName=DAN
AppVersion={#AppVersion}
AppPublisher=DAN
DefaultDirName={localappdata}\Programs\DAN
DefaultGroupName=DAN
DisableProgramGroupPage=yes
DisableDirPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename={#SetupName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
LicenseFile={#SourceDir}\LICENSE.txt
UninstallDisplayName=DAN
CloseApplications=force

[Tasks]
Name: desktopicon; Description: "Create desktop shortcuts"

[InstallDelete]
; An upgrade from a local-test build must not keep its local-network marker.
Type: files; Name: "{app}\config\local-network"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\DAN Node"; Filename: "powershell.exe"; \
    Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\DAN.ps1"" node"; \
    WorkingDir: "{app}"; IconFilename: "{app}\dan-provider.exe"; Comment: "Share this PC's GPU with DAN"
Name: "{group}\DAN Chat"; Filename: "powershell.exe"; \
    Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\DAN.ps1"" chat"; \
    WorkingDir: "{app}"; IconFilename: "{app}\dan-client.exe"; Comment: "Chat with a model on DAN"
Name: "{group}\Read me"; Filename: "{app}\README.txt"
Name: "{group}\Uninstall DAN"; Filename: "{uninstallexe}"
Name: "{userdesktop}\DAN Node"; Filename: "powershell.exe"; \
    Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\DAN.ps1"" node"; \
    WorkingDir: "{app}"; IconFilename: "{app}\dan-provider.exe"; Tasks: desktopicon
Name: "{userdesktop}\DAN Chat"; Filename: "powershell.exe"; \
    Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\DAN.ps1"" chat"; \
    WorkingDir: "{app}"; IconFilename: "{app}\dan-client.exe"; Tasks: desktopicon

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\DAN.ps1"" node"; \
    WorkingDir: "{app}"; Description: "Start DAN Node now"; Flags: postinstall nowait skipifsilent

[UninstallRun]
Filename: "taskkill.exe"; Parameters: "/F /T /IM dan-provider.exe"; Flags: runhidden; RunOnceId: "StopProvider"
Filename: "taskkill.exe"; Parameters: "/F /IM dan-sidecar.exe"; Flags: runhidden; RunOnceId: "StopSidecar"
Filename: "taskkill.exe"; Parameters: "/F /IM dan-client.exe"; Flags: runhidden; RunOnceId: "StopClient"
