package gmm

import java.text.SimpleDateFormat
import java.util.logging.{Formatter, LogManager, LogRecord, Logger}

/*
import org.apache.log4j.{LogManager => LogManager4J}
object Log2{
	val logger = LogManager4J.getLogger(this.getClass.toString)
}
*/

////Pure log, without logj4
object JulLogger{
	val sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
	val logManager = LogManager.getLogManager
	val logger = logManager.getLogger(Logger.GLOBAL_LOGGER_NAME)
	logManager.readConfiguration(this.getClass.getResourceAsStream("/jul.properties"))
}


class ConsoleFormatter extends Formatter {
	override def format(rcd: LogRecord): String = {
		//val src = Thread.currentThread.getStackTrace().find(x => x.getClassName+x.getMethodName == s"${rcd.getSourceClassName}${rcd.getSourceMethodName}")
		val src = Thread.currentThread.getStackTrace().find(x => x.getClassName == rcd.getSourceClassName).getOrElse("Log source not found").toString
		val lgl = f"${rcd.getLevel.toString.take(5)}%-5s"
		s"${JulLogger.sdf.format(System.currentTimeMillis)} $lgl [${Thread.currentThread.getName}] $src - ${rcd.getMessage}\n"
		//s"${Log.sdf.format(System.currentTimeMillis)} ${rcd.getLevel.toString.substring(0,4)} $src - ${rcd.getMessage}\n"
	}
}


class FileFormatter extends Formatter {
	override def format(rcd: LogRecord): String = {
		val src = Thread.currentThread.getStackTrace().find(x => x.getClassName == rcd.getSourceClassName).map(x =>s"${x.getFileName}:${x.getLineNumber}")
				.getOrElse("Log source not found")
		val lgl = f"${rcd.getLevel.toString.take(5)}%-5s"
		s"${JulLogger.sdf.format(System.currentTimeMillis)} $lgl $src - ${rcd.getMessage}\n"
	}
}
