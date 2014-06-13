module Utility

  # Global constants
  Version  = %w[1 2 7]
  Dscudadef = "-D__DSCUDA__=1 -D__DSCUDA_VERSION__=#{sprintf("0x%02x%02x%02x", Version[0], Version[1], Version[2])} "

  Dscudainfo = {
  }

  attr_writer :verbose

  # attr reader for @verbose
  def verbose
    if @verbose
      @verbose
    else
      verbose_default = 0
    end
  end

  def vinfo(*args)
    cname = (self.class.to_s =~ /Dscuda/) ? "#{self.class} : " : ''
    $stderr.puts("Info    : #{cname}#{args}") if 2 <= self.verbose 
  end

  def vwarn(*args)
    cname = (self.class.to_s =~ /Dscuda/) ? "#{self.class} : " : ''
    $stderr.puts("Warning : #{cname}#{args}") if 1 <= self.verbose
  end

  def vputs(lv, *args)
    $stderr.puts args if lv <= self.verbose 
  end

  def vprintf(lv, *args)
    $stderr.printf args if lv <= self.verbose 
  end

end # Utility

class String

  #
  # extract the outermost balanced token list.
  # separators can be any string other than double quote, such as ( and ), /* and */.
  # returns: matched_or_not, pre_matched_str, matched_str, post_matched_str
  # 
  # In the C grammer, double quote cannot be nested.
  # Therefor, in order to match a double-quoted string,
  # a simple regexp /\A " (?: [^"] | \\" )* " /xms will do.
  #
  def balanced_c_exp(lsep, rsep, strip=false)
    str = self
    lsep = Regexp.escape(lsep)
    rsep = Regexp.escape(rsep)
    #    puts "lsep:#{lsep}  rsep:#{rsep}  str:#{str}"

    # split into tokens (i.e. separators and other substrings).
    #
    tokens = []
    while not str.empty?
      case str
      when /\A " (?: [^"] | \\" )* " /xms
        str = $'
        t = $&
        if tokens[-1] and tokens[-1] !~ / \A (?: #{lsep} | #{rsep}) \z /xms
          tokens[-1].concat t
        else
          tokens.push t
        end
      when /\A #{lsep} /xms
        str = $'
        tokens.push $&
      when /\A #{rsep} /xms
        str = $'
        tokens.push $&
      when /\A . /xms
        str = $'
        t = $&
        if tokens[-1] and tokens[-1] !~ / \A (?: #{lsep} | #{rsep}) \z /xms
          tokens[-1].concat t
        else
          tokens.push t
        end
      else
        puts "no match."
        str = ''
      end
    end

#    puts "tokens : #{tokens.join(' | ')}"

    # parse the tokens.
    #
    matched  = false
    pre      = ''
    body     = ''
    post     = ''
    sepstack = []

    # pre match
    until tokens.empty?
      t = tokens.shift
      if t =~ / \A #{lsep} \z/xms
        body << t unless strip
        sepstack.push t
        break
      end
      pre << t
    end

    # body
    until tokens.empty?
      t = tokens.shift
      case t
      when / \A #{lsep} \z /xms
        body << t
        sepstack.push t
      when / \A #{rsep} \z /xms
        sepstack.pop
        if sepstack.empty?
          matched = true
          body << t unless strip
          break
        else
          body << t
        end
      else
        body << t
      end
    end

    post = tokens.join
    return matched, pre, body, post
  end

  def strip_balanced_c_exp(lsep, rsep)
    str = self.strip
    matched, pre, body, post = str.balanced_c_exp(lsep, rsep, false)
    if matched and body == str
      elsep = Regexp.escape(lsep)
      ersep = Regexp.escape(rsep)
      matched, pre, str, post = str.balanced_c_exp(lsep, rsep, true)
#      str = str.gsub(/\A #{elsep} (.*) #{ersep} \z/x) { $1 }
    end
    str
  end

  # 
  # split comma-separated C expressions into an Array.
  # eg) a, b, c(d, e), f => [a, b, c(d, e), f]
  #     (a, b, c(d, e)), f) => [(a, b, c(d, e)), f]
  #     a, b, c(d, e)f, g) => [a, b, c(d, e)f, g]
  # 
  def split_c_exp_list()
    srcstr = self
    dstarray = []
    while srcstr
      arg1, srcstr = get_1st_arg(srcstr)
      dstarray.push arg1
    end
    dstarray = dstarray.compact.collect{|v| v.strip}.select{|v| v.size > 0}
  end

  def get_1st_arg(srcstr)
    (ismatched, pre, body, post) = srcstr.balanced_c_exp('(', ')', false)
    if ismatched
      if pre =~ /\A([^,]*),(.*)\z/m # (exps) in the 2nd arg or later.
        car = $1
        cdr = $2 + body + post
      else # (exps) in the 1st arg.
        car_child, cdr_child = get_1st_arg(post)
        car = pre + body + car_child
        cdr = cdr_child
      end
    else
      if pre =~ /\A([^,]*),(.*)\z/m
        car = $1
        cdr = $2
      else
        car = pre
        cdr = nil
      end
    end
#     puts "car:#{car}  cdr:#{cdr}"
    return car, cdr
  end


  # 
  # remove C and C++ style comment.
  # comments inside a string literal, i.e., 
  # /*...*/ and // inside "...""
  # remain unchanged.
  # 
  # for example, the following 7-line input:
  # 
  # s = sin();
  # printf("this is /* not a comment */ "
  #        "but a string literal.\n");
  # printf("xx//this is not a comment but a string literal\n");
  # /* this is a C-style
  #    comment */
  # exit(1); // this is a C++ -style comment.
  # 
  # 
  # returns the following 6-line output:
  # 
  # s = sin();
  # printf("this is /* not a comment */ "
  #        "but a string literal.\n");
  # printf("xx//this is not a comment but a string literal\n");
  # 
  # exit(1); 
  # 

=begin
  #
  # simpler, but poor implementation using a regexp:
  #
  # this cannot handle // inside a string literal.
  # e.g.  "this is a // comment"  => "this is a 
  dststr = dststr.gsub(%r< // [^\n]*?\n >xms, "\n")
  dststr = dststr.gsub(%r<(/\*([^*]|\*[^/])*\*/)>xms, '')

  # this is better, not perfect, though.
  # e.g.  "this is a // comment"  => "this is a" 
  #       idealy the result should be "this is a // comment".
  dststr = dststr.gsub(
                       %r</\*[^*]*\*+([^/*][^*]*\*+)*/|//[^\n]*|("(\\.|[^"\\])*"|'(\\.|[^'\\])*'|.[^/"'\\]*)#defined $2 ? $2 : "";>se, '')
=end

  def omit_c_comments()
    srcstr = self
    dststr = ''
    until srcstr.empty?

      if srcstr !~ %r< (?: // [^\n]*?(?= \n)         | # C++ style comment
                                                       # note that should not eat the trailing newline.
                           /\*(?: [^*] | \*[^/] )* \*/ # C style comment.
                       )>xms
        dststr += srcstr
        break # no more comments.
      end

      strpre = $`
      strmatched = $&
      strpost = $'

      # a preceding literal string found. skip it.
      if strpre =~ / [^\\] " /xms
        dststr += $` + $&
        dststr = dststr[0..-2]
        literal = '"' + $' + strmatched + strpost
        if literal =~ /\A " (?: \\" | [^"] )*? " /xms
          dststr += $&
          srcstr = $'
        else
          raise "unmatched double quotation found: #{literal}. just a typo?"
        end
        next
      end
      dststr += strpre
      srcstr = strpost
    end # until

    return dststr
  end

  # double quote srcword, if not yet double quoted.
  def double_quote
    srcword = self.dup
    case srcword
    when /\A " [^"]* " \z/xs # already quoted.
      dstword = srcword
    when /\A & \s* (\w+) \z/xs
      dstword = %Q<"#{$1}">
    else
      dstword = %Q<"#{srcword}">
    end
    dstword
  end

  #
  # As the 1st returning value, this method returns
  # a literal string and anything preceding it, if any.
  # If the literal is not terminated, eat letters from 'poststr'
  # until a terminator (i.e. a double quotation) found.
  # Unconsumed 'poststr' is returned as the 2nd returning value.
  # Two null strings are returned, if no literal found in 'self'.
  #
  # eg) 'this is "a literal st'.skip_a_literal_string('ring" crossing over two vars')
  #     => 'this is "a literal string"', ' crossing over two vars'
  #
  def eat_a_c_literal_string(poststr)
    str = self.dup
    dststr = ''
    leftstr = ''
    if str =~ / [^\\] " /xms
      dststr += $` + $&
      dststr = dststr[0..-2]
      literal = '"' + $' + poststr

      if literal =~ /\A " (?: \\" | [^"] )*? " /xms
        dststr += $&
        leftstr = $'
      else
        raise "unbalanced double quotation found: #{literal}. just a typo?"
      end
    end
    return dststr, leftstr
  end

  def eat_a_c_literal
    prestr = self.dup
    literal = ''
    leftstr = ''
    if prestr =~ / ([^\\]?) " /xms
      prestr = $`+ $1
      literal = '"' + $'



      if literal =~ /\A " (?: \\" | [^"] )*? " /xms
        literal = $&
        leftstr = $'
      else
        raise "unbalanced double quotation found: #{literal}. just a typo?"
      end
    end
    return prestr, literal, leftstr
  end

end # String
