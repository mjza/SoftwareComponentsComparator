# DB Port forwarding

To establish an SSH port forwarding tunnel on a Mac, you'll use the ssh -L command in Terminal. This command creates a tunnel that allows you to connect to a remote server's service, making it appear as if it's running locally on your Mac. 

## Here's how to do it: 

1. Open Terminal: Launch the Terminal application on your Mac.
2. Run the command: Use the following syntax, replacing the placeholders with your specific values:

```bash
    ssh -L <local-port>:<remote-host>:<remote-port> <remote-user>@<remote-host>
```

<local-port>: The port you want to use on your Mac to access the service. 
<remote-host>: The hostname or IP address of the remote server. 
<remote-port>: The port the service you want to access is running on the remote server. 
<remote-user>: The username you use to log in to the remote server. 

Example: If you want to access a database service (port 5432) on a remote server with the hostname mc49 using your local port 15432, and your ssh username is uniuser, the command would be: 

```
ssh -L 15432:mc97:5432 uniuser@arc.ucalgary.ca
```