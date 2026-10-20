Option Explicit

Dim fso, shell, baseDir, venvDir, pythonExe, backendDir
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

baseDir = fso.GetParentFolderName(WScript.ScriptFullName)
venvDir = fso.GetParentFolderName(baseDir)
pythonExe = venvDir & "\Scripts\python.exe"
backendDir = baseDir & "\backend"

If Not fso.FileExists(pythonExe) Then
    MsgBox "未找到 Python 虚拟环境: " & pythonExe & vbCrLf & "请先配置项目环境。", 16, "新闻摘要系统"
    WScript.Quit 1
End If

' 检查前端 dist 是否存在
If Not fso.FolderExists(baseDir & "\frontend\dist") Then
    MsgBox "未找到前端构建产物 frontend\dist，请先运行 npm run build", 16, "新闻摘要系统"
    WScript.Quit 1
End If

' 1. 后台启动后端（uvicorn，端口 8000，同时托管前端页面）
shell.CurrentDirectory = backendDir
shell.Run """" & pythonExe & """ -m uvicorn app.main:app --host 127.0.0.1 --port 8000", 0, False

' 2. 等待服务就绪（最多约 60 秒）
Dim http, ok, waited
waited = 0
ok = False
Do While waited < 60 And Not ok
    WScript.Sleep 2000
    waited = waited + 2
    ok = Probe("http://127.0.0.1:8000/api/health")
Loop

If Not ok Then
    MsgBox "服务启动超时，请检查后端是否正常", 16, "新闻摘要系统"
    WScript.Quit 1
End If

' 3. 打开浏览器访问系统
shell.Run "http://127.0.0.1:8000/"

Function Probe(url)
    On Error Resume Next
    Set http = CreateObject("MSXML2.ServerXMLHTTP")
    http.setTimeouts 1500, 1500, 1500, 1500
    http.Open "GET", url, False
    http.Send
    If Err.Number = 0 And http.Status = 200 Then
        Probe = True
    Else
        Probe = False
    End If
    On Error Goto 0
End Function