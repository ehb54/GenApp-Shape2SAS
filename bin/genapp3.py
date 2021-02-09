"""
GenApp helper library
"""

import sys
import json
import socket
import pickle
#import cStringIO

class genapp(object):

    def __init__( self, jsoninput ):
        """Always initialize with json input either as a json string or as a dict or object"""
        if isinstance( jsoninput, dict ) :
            self.jsoninput = jsoninput
        else:
            if isinstance( jsoninput, basestring ) :
                try:
                    self.jsoninput = json.loads( jsoninput )
                except ValueError:
                    raise Exception( 'jsoninput decoding error: malformed json string' )
            else:
                raise Exception( 'GenApp must be initialized with jsoninput either as a json string or as a dict' )

        if not '_uuid' in self.jsoninput:
            raise Exception( 'jsoninput must contain key "_uuid"' )

        self.udp_enabled  = '_udphost' in self.jsoninput and '_udpport' in self.jsoninput
        self.tcp_enabled  = '_tcphost' in self.jsoninput and '_tcpport' in self.jsoninput
        self.tcpr_enabled = self.tcp_enabled and '_tcprport' in self.jsoninput
        self.mpl_enabled  = '_mplhost' in self.jsoninput
        # if mpl_enabled, check mpl plot ports and interval timer keepalive

    def info( self ):
        return {
            'udp_enabled'   : self.udp_enabled
            ,'tcp_enabled'  : self.tcp_enabled
            ,'tcpr_enabled' : self.tcpr_enabled
            ,'mpl_enabled'  : self.mpl_enabled
            ,'_uuid'        : self.jsoninput['_uuid']
        }

    def tcpquestion( self, question, timeout=300, buffersize=65536 ):
        # question is either a dict or json string
        # timeout is in seconds
        # buffersize is for the answer, so if you expect larger than 64k of total json string size, use a larger number

        if not self.tcpr_enabled:
            return { 'error':'no tcp support' }

        # build question

        msg = {
            '_uuid'    : self.jsoninput['_uuid']
            ,'timeout' : timeout
        }

        if isinstance( question, basestring ):
            try:
                msg['_question'] = json.loads(question)
            except ValueError:
                return {'error':'json question decoding error'}
        elif isinstance( question, dict ):
            msg['_question'] = question
        else:
            return {'error':'question must be a json string or dict'}

        msgj = json.dumps(msg)
        # a newline is also required when sending a question
        msgj += '\n'

        # send question
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.jsoninput['_tcphost'],int( self.jsoninput['_tcpport']) ))
        s.send(msgj.encode('utf-8'))

        # receive answer

        data = s.recv(buffersize)
        s.close()
        return json.loads(data)

    def tcpmessagebox( self, message ):
        """send a message box over tcp"""

        if not self.tcp_enabled:
            return { 'error':'no tcp support' }

        msg = {
            '_uuid'    : self.jsoninput['_uuid']
        }

        if isinstance( message, basestring ):
            try:
                msg['_message'] = json.loads(message)
            except ValueError:
                return {'error':'tcpmessage:json message decoding error'}
        elif isinstance( message, dict ):
            msg['_message'] = message
        else:
            return {'error':'message must be a json string or dict'}


        # send question
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.jsoninput['_tcphost'],int( self.jsoninput['_tcpport']) ))
        s.send(json.dumps(msg).encode('utf-8'))
        s.close()
        return {'status':'ok'}

    def tcpmessage( self, message ):
        """send a message over tcp"""

        if not self.tcp_enabled:
            return { 'error':'no tcp support' }

        if isinstance( message, basestring ):
            try:
                msg = json.loads(message)
            except ValueError:
                return {'error':'tcpmessage:json message decoding error'}
        elif isinstance( message, dict ):
            msg = message
        else:
            return {'error':'message must be a json string or dict'}

        msg[ '_uuid' ] = self.jsoninput['_uuid']

        # send question
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.jsoninput['_tcphost'],int( self.jsoninput['_tcpport']) ))
        s.send(json.dumps(msg).encode('utf-8'))
        s.close()
        return {'status':'ok'}

    def udpmessagebox( self, message ):
        """send a message box over udp"""

        if not self.udp_enabled:
            return { 'error':'no udp support' }

        msg = {
            '_uuid'    : self.jsoninput['_uuid']
        }

        if isinstance( message, str ):
            try:
                msg['_message'] = json.loads(message)
            except ValueError:
                return {'error':'udpmessage:json message decoding error'}
        elif isinstance( message, dict ):
            msg['_message'] = message
        else:
            return {'error':'message must be a json string or dict'}

        # send message
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto( json.dumps( msg ).encode('utf-8'), ( self.jsoninput['_udphost'], int( self.jsoninput['_udpport'] ) ) )
        return {'status':'ok'}

    def udpmessage( self, message ):
        """send a message over udp"""

        if not self.udp_enabled:
            return { 'error':'no udp support' }

        msg = {
            '_uuid'    : self.jsoninput['_uuid']
        }

        if isinstance( message, str ):
            try:
                msg = json.loads(message)
            except ValueError:
                return {'error':'udpmessage:json message decoding error'}
        elif isinstance( message, dict ):
            msg = message
        else:
            return {'error':'message must be a json string or dict'}

        msg[ '_uuid' ] = self.jsoninput['_uuid']

        # send message
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto( json.dumps( msg ).encode('utf-8'), ( self.jsoninput['_udphost'], int( self.jsoninput['_udpport'] ) ) )
        return {'status':'ok'}
    
    # extend plotshow with figure id which should be in the jsoninput
    # the jsoninput figure id should have an assigned port, which we will use

    def plotshow( self, mpl, plt, port ):
        """Show a plot for matplotlib via GenApp UI on the defined host"""

        if not self.mpl_enabled:
            return { 'error':'no mpl support' }

        mpl.rcParams['webagg.open_in_browser'] = False
        mpl.rcParams['webagg.address'] = "0.0.0.0"
        mpl.rcParams['webagg.port'] = port
        # this should be pushed to backend_webagg.py right before server starts
        # perhaps we could redefine the class member externally?
        print("now messsage that plot (will shortly be) available")
#        self.stdout_off()
        plt.show()
#        self.stdout_on()
        print("register atend pickler doesn't seem relevant, pickle here")
        pickle.dump( plt.figure(), file( 'plot-' + str( port ) + '.pickle' , 'w' ) )
        print("pickle saved")

#    @staticmethod
#    def stdout_off():
#        sys.stdout = cStringIO.StringIO()

#    @staticmethod
#    def stdout_on():
#        sys.stdout.close()
#        sys.stdout = sys.__stdout__

    @staticmethod
    def test():
        ga = genapp( {
            '_uuid'     : 'my_uuid'
            ,'_udphost' : '127.0.0.1'
            ,'_udpport'  : 2234
            ,'_mplhost'  : '127.0.0.1'
        } )

        print(json.dumps( ga.info() ))
