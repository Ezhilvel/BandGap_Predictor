<html >
<title> Band Gap Predictor </title>
<head> </head>
<body>
<form method = "post" action = "<?php echo htmlspecialchars($_SERVER["PHP_SELF"]);?>">
<input type = text name = 'ele'> </input>
<input type = "submit" name = 'but' value = "submit" />
</form>
</body>
</html>
<?php
if(isset($_POST['but']))
{
    if(!empty($_POST['ele']))
    {
     $ele =$_POST['ele'];
     $command = 'python index.py '.$ele;
     exec($command, $out, $status);
     print_r($out);
    }
    else {
	echo"<script>alert('enter the binary compound');</script>";
	 }
}
?>
